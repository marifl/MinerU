"""`mineru-de`: MinerU 4.x parsing behind the MinerU 3.x command line and output layout.

    mineru-de -p <file|dir> -o <out> [-m auto|txt|ocr] [-b BACKEND] [-l LANG] [-s N] [-e N]
              [-f true|false] [-t true|false] [--image-analysis true|false]
              [--effort medium|high] [--title-levels auto|slides|numbered|off] [--llm off|local|cloud]

Output per input, as MinerU 3.x wrote it: <out>/<stem>/<backend dir>/ with
<stem>.md, <stem>_content_list.json (plain text like 3.x), <stem>_content_list_v2.json (with
bold/italic as `style`), images/, <stem>_origin.pdf,
plus <stem>_middle_v4.json (MinerU 4 schema), <stem>_mineru.json (provenance) and
<stem>_pruefung.json (acceptance report: ok, incomplete, failed or unchecked, with the pages and
words that are missing). With --gate fail (default) an unproven document ends with exit code 1.
Missing lines of an incomplete page are filled in from the PDF text layer and marked with
"source": "text_layer_repair" (--repair off to keep the output exactly as MinerU produced it).

There is deliberately no <stem>_middle.json: MinerU 4 has a different middle schema, and readers of
the 3.x `pdf_info` layout should fail loudly instead of reading the wrong structure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import pypdfium2 as pdfium
from docvortex.schema import TextSpan
from pydantic import BaseModel

from mineru.filetypes import is_flash_only_parse_extension
from mineru.kit.common import ensure_supported_inputs, expand_input_paths
from mineru.parser import parse
from mineru.parser.base import ParseResult
from mineru.parser.writer import FileBasedDataWriter
from mineru.render.content_list import render_content_list
from mineru.render.content_list_v2 import render_content_list_v2
from mineru.types import MiddleJson
from mineru.version import __version__ as mineru_version

from . import __version__
from .gate import MIN_PAGE_RECALL, check_document
from .llm_titles import apply_llm_levels, load_profiles
from .repair import repair_pages
from .title_levels import apply_title_levels

# 3.x backend name -> (kind, 3.x output directory name; {method} is the parse method).
# MinerU 4 has tiers instead of backends: pipeline and hybrid with --effort medium map to "basic",
# hybrid with --effort high and every vlm backend to "standard" (upstream's own mapping).
BACKENDS = {
    "pipeline": ("pipeline", "{method}"),
    "hybrid-engine": ("hybrid", "hybrid_{method}"),
    "hybrid-http-client": ("hybrid", "hybrid_{method}"),
    "hybrid-auto-engine": ("hybrid", "hybrid_{method}"),  # name used by MinerU 3.1-3.3
    "vlm-engine": ("vlm", "vlm"),
    "vlm-http-client": ("vlm", "vlm"),
    "vlm-auto-engine": ("vlm", "vlm"),
    "vlm-mlx-engine": ("vlm", "vlm"),
}
EFFORT_TIER = {"medium": "basic", "high": "standard"}


def _bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true or false, got {value!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mineru-de", description=__doc__.split("\n\n")[0])
    parser.add_argument("-p", "--path", required=True, help="input file or directory")
    parser.add_argument("-o", "--output", required=True, help="output directory")
    parser.add_argument("-m", "--method", choices=["auto", "txt", "ocr"], default="auto")
    parser.add_argument("-b", "--backend", choices=sorted(BACKENDS), default="hybrid-engine")
    parser.add_argument("--effort", choices=["medium", "high"], default="medium",
                        help="hybrid backends: medium -> MinerU 4 tier basic, high -> tier standard")
    parser.add_argument("-l", "--lang", default=None, help="accepted for 3.x compatibility; MinerU 4 has no language switch")
    parser.add_argument("-s", "--start", type=int, default=None, help="first page, 0-based")
    parser.add_argument("-e", "--end", type=int, default=None, help="last page, 0-based")
    parser.add_argument("-f", "--formula", type=_bool, default=True)
    parser.add_argument("-t", "--table", type=_bool, default=True)
    parser.add_argument("--image-analysis", type=_bool, default=True)
    parser.add_argument("--title-levels", choices=["auto", "slides", "numbered", "off"], default="auto")
    parser.add_argument("--llm", default="off",
                        help="LLM profile for the headings the rules leave open: off (default), local, cloud, "
                             "or a profile from $MINERU_HOME/ifland-llm.json")
    parser.add_argument("--repair", choices=["text-layer", "off"], default="text-layer",
                        help="text-layer: append the missing lines of an incomplete page from the PDF text "
                             "layer, marked with source=text_layer_repair (default)")
    parser.add_argument("--gate", choices=["fail", "warn", "off"], default="fail",
                        help="fail: exit non-zero when a document is not proven complete (default)")
    parser.add_argument("--min-page-recall", type=float, default=MIN_PAGE_RECALL,
                        help="share of a page's text-layer words that must appear in the output")
    return parser


def _page_range(start: int | None, end: int | None) -> str:
    if start is None and end is None:
        return ""
    first = (start or 0) + 1
    return f"{first}-{end + 1}" if end is not None else f"{first}-r1"


def _is_landscape(path: Path) -> bool:
    if path.suffix.lower() != ".pdf":
        return False
    doc = pdfium.PdfDocument(str(path))
    try:
        landscape = sum(1 for page in doc if page.get_width() > page.get_height())
        return landscape > len(doc) / 2
    finally:
        doc.close()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dump(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=4), encoding="utf-8")


def _without_styles(middle_json: MiddleJson) -> MiddleJson:
    """Deep copy of `middle_json` with every inline text style removed."""
    plain = middle_json.model_copy(deep=True)
    pending: list[object] = [plain]
    while pending:
        node = pending.pop()
        if isinstance(node, TextSpan):
            node.styles = []
        elif isinstance(node, BaseModel):
            pending.extend(getattr(node, name) for name in type(node).model_fields)
        elif isinstance(node, (list, tuple)):
            pending.extend(node)
    return plain


def write_legacy_layout(result: ParseResult, source: Path, target: Path, provenance: dict) -> None:
    """Write `result` into `target` with MinerU 3.x file names."""
    stem = source.stem
    target.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target) as tmp_name:
        tmp = Path(tmp_name)
        result.save(FileBasedDataWriter(str(tmp)))
        exported = ParseResult.from_json((tmp / "middle_json.json").read_text(encoding="utf-8")).middle_json

        if (target / "images").exists():
            shutil.rmtree(target / "images")
        if (tmp / "images").exists():
            shutil.move(tmp / "images", target / "images")
        shutil.move(tmp / "markdown.md", target / f"{stem}.md")
        shutil.move(tmp / "middle_json.json", target / f"{stem}_middle_v4.json")
    # 3.x content lists carried plain text; bold/italic stay in content_list_v2 (`style`) and markdown
    _dump(target / f"{stem}_content_list.json", render_content_list(_without_styles(exported)))
    _dump(target / f"{stem}_content_list_v2.json", render_content_list_v2(exported))
    if source.suffix.lower() == ".pdf":
        shutil.copyfile(source, target / f"{stem}_origin.pdf")
    _dump(target / f"{stem}_mineru.json", provenance)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.formula or not args.table:
        sys.exit("mineru-de: MinerU 4 cannot switch off formula or table recognition (-f/-t false)")

    kind, dir_pattern = BACKENDS[args.backend]
    tier = {"pipeline": "basic", "vlm": "standard"}.get(kind) or EFFORT_TIER[args.effort]
    page_range = _page_range(args.start, args.end)
    output = Path(args.output).expanduser()
    paths = expand_input_paths([args.path])
    ensure_supported_inputs(paths)
    not_ok: list[tuple[Path, dict]] = []

    for source in paths:
        source_tier = "flash" if is_flash_only_parse_extension(source) else tier
        result = parse(
            source,
            tier=source_tier,
            ocr_mode=args.method,
            image_analysis=args.image_analysis,
            page_range=page_range,
        )
        mode = args.title_levels
        if mode == "auto" and source_tier == "flash":
            mode = "off"  # Office, EPUB and HTML carry their own heading structure
        applied = apply_title_levels(result.middle_json, mode, landscape=_is_landscape(source))
        llm = apply_llm_levels(result.middle_json, applied, args.llm) if args.llm != "off" else None
        target = output / source.stem / dir_pattern.format(method=args.method)
        provenance = {
            "tool": {"mineru_de": __version__, "mineru": mineru_version},
            "argv": sys.argv[1:] if argv is None else argv,
            "source": {"path": str(source.resolve()), "sha256": _sha256(source)},
            "parse": {
                "backend": args.backend,
                "effort": args.effort,
                "tier": source_tier,
                "ocr_mode": args.method,
                "image_analysis": args.image_analysis,
                "page_range": page_range or "all",
            },
            "title_levels": {"requested": args.title_levels, "applied": applied, "llm": llm and llm.as_dict()},
            "ignored": {"lang": args.lang} if args.lang else {},
        }
        write_legacy_layout(result, source, target, provenance)

        if args.gate == "off":
            report = {"source": str(source), "status": "unchecked", "reason": "gate switched off"}
        else:
            content_list_path = target / f"{source.stem}_content_list.json"
            content_list = json.loads(content_list_path.read_text(encoding="utf-8"))
            checked = check_document(content_list, source, target, min_page_recall=args.min_page_recall)
            repaired: list[dict] = []
            if args.repair == "text-layer" and checked["incomplete_pages"]:
                repaired = repair_pages(content_list, source, [p["page"] for p in checked["incomplete_pages"]])
                if repaired:
                    _dump(content_list_path, content_list)
                    checked = check_document(content_list, source, target, min_page_recall=args.min_page_recall)
            report = {"source": str(source), **checked, "repaired_pages": repaired}
        _dump(target / f"{source.stem}_pruefung.json", report)
        if report["status"] != "ok":
            not_ok.append((source, report))
        print(f"mineru-de: {source} -> {target} [{report['status']}]", file=sys.stderr)

    for source, report in not_ok:
        pages = ", ".join(f"S.{p['page']} {p['recall']:.0%}" for p in report.get("incomplete_pages", [])[:5])
        problems = ", ".join(p["problem"] for p in report.get("problems", []))
        print(f"mineru-de: {source.name}: {report['status']} {pages} {problems}".rstrip(), file=sys.stderr)
    return 1 if (args.gate == "fail" and any(r["status"] in {"failed", "incomplete"} for _, r in not_ok)) else 0


if __name__ == "__main__":
    sys.exit(main())
