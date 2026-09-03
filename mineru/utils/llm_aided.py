# Copyright (c) Opendatalab. All rights reserved.
import importlib.util
import os
import re
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, partial

import json_repair
from loguru import logger
from openai import OpenAI

from mineru.backend.pipeline.pipeline_middle_json_mkcontent import merge_para_with_text
from mineru.utils.enum_class import BlockType


TITLE_BLOCK_TYPES = {
    BlockType.TITLE,
    BlockType.DOC_TITLE,
    BlockType.PARAGRAPH_TITLE,
}
MAX_TITLE_GROUP_WORKERS = 4

# Public names a user override module may define to replace the built-in
# prompt builders. Keyed by the built-in function name.
PROMPT_OVERRIDE_NAMES = {
    "_build_title_optimize_prompt": "build_title_optimize_prompt",
    "_build_relative_title_optimize_prompt": "build_relative_title_optimize_prompt",
    "_build_chunk_title_optimize_prompt": "build_chunk_title_optimize_prompt",
}

# Headings that look like chapter starts; chunk boundaries prefer to sit right before them.
CHUNK_ANCHOR_PATTERN = re.compile(
    r"^\s*(\d{1,3}\s+\S|(kapitel|chapter|teil|part|abschnitt|section)\s+[\divxlc]+\b|[IVXLC]{1,6}\.\s)",
    re.IGNORECASE,
)
DEFAULT_CHUNK_CONTEXT = 8
NUMBERED_HEADING_PATTERN = re.compile(r"^\s*(\d+(?:\.\d+)*)\s+\S")
MAX_TITLE_LEVEL = 4
MIN_NUMBERING_VOTES = 5


@lru_cache(maxsize=None)
def _load_prompt_override(override_file):
    path = os.path.expanduser(override_file)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"llm-aided-config.title_aided.override_file does not exist: {path}"
        )
    spec = importlib.util.spec_from_file_location("mineru_llm_aided_override", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_prompt_builder(title_aided_config, prompt_builder):
    """Return the user-provided prompt builder for `prompt_builder` if
    `override_file` is configured and defines one, else `prompt_builder`."""
    override_file = title_aided_config.get("override_file")
    if not override_file:
        return prompt_builder
    override = _load_prompt_override(override_file)
    public_name = PROMPT_OVERRIDE_NAMES.get(getattr(prompt_builder, "__name__", None))
    if public_name is None:
        return prompt_builder
    custom = getattr(override, public_name, None)
    if custom is None:
        return prompt_builder
    if not callable(custom):
        raise TypeError(f"{override_file}: {public_name} must be callable")
    return custom


def _get_title_line_avg_height(block):
    line_avg_height = block.get("line_avg_height")
    if isinstance(line_avg_height, (int, float)) and line_avg_height > 0:
        return line_avg_height

    title_block_line_height_list = []
    for line in block.get("lines", []):
        # 标题行高是 LLM prompt 的几何提示，这里只消费有效 bbox，不回写 middle json。
        bbox = line.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        line_height = bbox[3] - bbox[1]
        if line_height > 0:
            title_block_line_height_list.append(int(line_height))

    if len(title_block_line_height_list) > 0:
        return sum(title_block_line_height_list) / len(title_block_line_height_list)

    bbox = block.get("bbox")
    if bbox and len(bbox) >= 4:
        return max(0, int(bbox[3] - bbox[1]))
    return 0


def _collect_title_block_refs(page_info_list):
    title_block_refs = []
    title_types = set()

    for page_info in page_info_list:
        for block in page_info.get("para_blocks", []):
            block_type = block.get("type")
            if block_type in TITLE_BLOCK_TYPES:
                title_block_refs.append((page_info, block))
                title_types.add(block_type)

    return title_block_refs, title_types


def _build_title_dict(title_block_refs):
    title_dict = {}

    for i, (page_info, block) in enumerate(title_block_refs):
        title_dict[str(i)] = [
            merge_para_with_text(block),
            _get_title_line_avg_height(block),
            int(page_info["page_idx"]) + 1,
        ]

    return title_dict


def _build_title_optimize_prompt(title_dict):
    return f"""输入的内容是一篇文档中所有标题组成的字典，请根据以下指南优化标题的结果，使结果符合正常文档的层次结构：

1. 字典中每个value均为一个list，包含以下元素：
    - 标题文本
    - 文本行高是标题所在块的平均行高
    - 标题所在的页码

2. 保留原始内容：
    - 输入的字典中所有元素都是有效的，不能删除字典中的任何元素
    - 请务必保证输出的字典中元素的数量和输入的数量一致

3. 保持字典内key-value的对应关系不变

4. 优化层次结构：
    - 根据标题内容的语义为每个标题元素添加适当的层次结构
    - 行高较大的标题一般是更高级别的标题
    - 标题从前至后的层级必须是连续的，不能跳过层级
    - 标题层级最多为4级，不要添加过多的层级
    - 优化后的标题只保留代表该标题的层级的整数，不要保留其他信息

5. 合理性检查与微调：
    - 在完成初步分级后，仔细检查分级结果的合理性
    - 根据上下文关系和逻辑顺序，对不合理的分级进行微调
    - 确保最终的分级结果符合文档的实际结构和逻辑

IMPORTANT:
请直接返回优化过的由标题层级组成的字典，格式为{{标题id:标题层级}}，如下：
{{
  0:1,
  1:2,
  2:2,
  3:3
}}
不需要对字典格式化，不需要返回任何其他信息。

Input title list:
{title_dict}

Corrected title list:
"""


def _build_relative_title_optimize_prompt(title_dict):
    return f"""输入内容是某一篇文档中除文章标题外的全部章节/段落标题组成的字典。

请注意：
- 文章标题不在本次输入中，已经由系统单独识别并设置为1级标题

1. 字典中每个value均为一个list，包含以下元素：
    - 标题文本
    - 文本行高是标题所在块的平均行高
    - 标题所在的页码

2. 保留原始内容：
    - 输入的字典中所有元素都是有效的，不能删除字典中的任何元素
    - 请务必保证输出的字典中元素的数量和输入的数量一致

3. 保持字典内key-value的对应关系不变

4. 优化层次结构：
    - 根据标题内容的语义为每个标题元素添加适当的层次结构
    - 行高较大的标题一般是更高级别的标题
    - 标题从前至后的层级必须是连续的，不能跳过层级
    - 标题层级最多为4级，不要添加过多的层级
    - 优化后的标题只保留代表该标题的层级的整数，不要保留其他信息

5. 合理性检查与微调：
    - 在完成初步分级后，仔细检查分级结果的合理性
    - 根据上下文关系和逻辑顺序，对不合理的分级进行微调
    - 确保最终的分级结果符合文档的实际结构和逻辑

IMPORTANT:
请直接返回优化后的标题层级字典，格式为{{标题id:标题层级}}，如下：
{{
  0:1,
  1:2,
  2:2,
  3:3
}}
不要返回 Markdown，不要返回代码块，不要返回任何解释文字。

Input title list:
{title_dict}

Corrected title list:
"""


def _build_chunk_title_optimize_prompt(context, title_dict):
    return f"""输入内容是一篇长文档中一段连续的标题，该文档的标题正在分批处理，本批之前的标题已经完成分级。

已分级的上下文（格式为 [标题文本, 行高, 页码, 层级]），其中 "path" 是当前仍然打开的各级章节，"previous" 是紧邻本批之前的若干标题：
{context}

请注意：
- 上下文中的标题不在本次输入中，不要重新输出它们
- 本批标题的层级必须与上下文保持一致：同一类标题使用同一层级，位于上下文章节之下的标题层级要更深

1. 字典中每个value均为一个list，包含以下元素：
    - 标题文本
    - 文本行高是标题所在块的平均行高
    - 标题所在的页码

2. 保留原始内容：
    - 输入的字典中所有元素都是有效的，不能删除字典中的任何元素
    - 请务必保证输出的字典中元素的数量和输入的数量一致

3. 保持字典内key-value的对应关系不变

4. 优化层次结构：
    - 根据标题内容的语义为每个标题元素添加适当的层次结构
    - 行高较大的标题一般是更高级别的标题
    - 标题层级最多为4级，不要添加过多的层级
    - 优化后的标题只保留代表该标题的层级的整数，不要保留其他信息

IMPORTANT:
请直接返回优化后的标题层级字典，格式为{{标题id:标题层级}}，如下：
{{
  0:1,
  1:2,
  2:2,
  3:3
}}
不要返回 Markdown，不要返回代码块，不要返回任何解释文字。

Input title list:
{title_dict}

Corrected title list:
"""


def _request_title_levels(title_aided_config, title_dict, prompt_builder=None):
    if len(title_dict) == 0:
        return {}

    client = OpenAI(
        api_key=title_aided_config["api_key"],
        base_url=title_aided_config["base_url"],
    )

    retry_count = 0
    max_retries = 3
    expected_keys = set(range(len(title_dict)))
    if prompt_builder is None:
        prompt_builder = _build_title_optimize_prompt
    prompt_builder = _resolve_prompt_builder(title_aided_config, prompt_builder)
    title_optimize_prompt = prompt_builder(title_dict)

    logger.debug(f"Requesting LLM for title optimization with prompt: {title_optimize_prompt}")

    api_params = {
        "model": title_aided_config["model"],
        "messages": [{"role": "user", "content": title_optimize_prompt}],
        "temperature": 0.7,
        "stream": True,
    }
    if "enable_thinking" in title_aided_config:
        api_params["extra_body"] = {
            "enable_thinking": title_aided_config["enable_thinking"]
        }
    extra_body = title_aided_config.get("extra_body")
    if extra_body:
        if not isinstance(extra_body, dict):
            raise TypeError("llm-aided-config.title_aided.extra_body must be a JSON object")
        api_params["extra_body"] = {**api_params.get("extra_body", {}), **extra_body}

    while retry_count < max_retries:
        try:
            completion = client.chat.completions.create(**api_params)
            content_pieces = []
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    content_pieces.append(chunk.choices[0].delta.content)

            content = "".join(content_pieces).strip()
            if "</think>" in content:
                idx = content.index("</think>") + len("</think>")
                content = content[idx:].strip()

            logger.debug(f"Raw LLM output for title levels: {content}")
            dict_completion = json_repair.loads(content)
            dict_completion = {int(k): int(v) for k, v in dict_completion.items()}

            if set(dict_completion.keys()) == expected_keys:
                return dict_completion

            logger.warning(
                "The keys in the optimized title result do not match the input titles."
            )
        except Exception as e:
            logger.exception(e)

        retry_count += 1

    logger.error("Failed to decode dict after maximum retries.")
    return None


def _apply_levels_to_blocks(title_block_refs, levels_by_index):
    if levels_by_index is None:
        return

    for i, (_, block) in enumerate(title_block_refs):
        block["level"] = int(levels_by_index[i])


def _normalize_title_types(title_block_refs):
    for _, block in title_block_refs:
        if block.get("type") in [BlockType.DOC_TITLE, BlockType.PARAGRAPH_TITLE]:
            block["type"] = BlockType.TITLE


def _get_title_block_identity(block):
    block_index = block.get("index")
    if block_index is not None:
        return ("index", block_index)

    return (
        "bbox_text",
        tuple(block.get("bbox", [])),
        merge_para_with_text(block),
    )


def _sync_para_titles_to_preproc(page_info_list):
    for page_info in page_info_list:
        para_title_map = {}
        for block in page_info.get("para_blocks", []):
            if block.get("type") in TITLE_BLOCK_TYPES:
                para_title_map[_get_title_block_identity(block)] = block

        if len(para_title_map) == 0:
            continue

        for block in page_info.get("preproc_blocks", []):
            if block.get("type") not in TITLE_BLOCK_TYPES:
                continue

            para_block = para_title_map.get(_get_title_block_identity(block))
            if para_block is None:
                continue

            block["type"] = para_block.get("type", block.get("type"))
            if "level" in para_block:
                block["level"] = para_block["level"]


def _run_single_pass_title_leveling(title_block_refs, title_aided_config):
    title_dict = _build_title_dict(title_block_refs)
    levels_by_index = _request_title_levels(title_aided_config, title_dict)
    _apply_levels_to_blocks(title_block_refs, levels_by_index)


def _split_title_chunks(title_block_refs, chunk_size):
    """Split refs into consecutive chunks of at most chunk_size titles. A chunk ends early
    if a chapter-like anchor (numbering pattern or a line height in the top decile) lies in
    its last 40%, so chapters stay together whenever they are shorter than chunk_size."""
    total = len(title_block_refs)
    if total <= chunk_size:
        return [title_block_refs]

    entries = [
        (merge_para_with_text(block), _get_title_line_avg_height(block))
        for _, block in title_block_refs
    ]
    heights = sorted(h for _, h in entries if isinstance(h, (int, float)) and h > 0)
    height_threshold = heights[int(len(heights) * 0.9)] if heights else float("inf")
    anchors = {
        i for i, (text, height) in enumerate(entries)
        if CHUNK_ANCHOR_PATTERN.match(text) or (height is not None and height >= height_threshold)
    }

    chunks = []
    start = 0
    while start < total:
        end = min(start + chunk_size, total)
        if end < total:
            lowest_cut = start + max(1, int(chunk_size * 0.6))
            anchor_cut = next((i for i in range(end, lowest_cut, -1) if i in anchors), None)
            if anchor_cut is not None:
                end = anchor_cut
        chunks.append(title_block_refs[start:end])
        start = end
    return chunks


def _build_chunk_context(leveled_entries, context_size):
    if not leveled_entries:
        return None
    path = {}
    for entry in leveled_entries:
        path[entry[3]] = entry
    for level in list(path):
        # a section is still open only if no later heading sits at a higher level
        later_higher = any(e[3] < level for e in leveled_entries[leveled_entries.index(path[level]) + 1:])
        if later_higher:
            del path[level]
    return {
        "path": [path[level] for level in sorted(path)],
        "previous": leveled_entries[-context_size:],
    }


def _numbering_depth(text):
    match = NUMBERED_HEADING_PATTERN.match(text)
    return match.group(1).count(".") + 1 if match else None


def _realign_chunks_by_numbering(chunk_levels):
    """chunk_levels: list of lists of (text, level|None) per chunk, mutated in place.
    Dotted numbering ("3.2", "3.2.1") is ground truth for relative depth. A chunk whose numbered
    titles all sit a constant offset away from the document-wide majority level for their depth is
    shifted back as a whole; afterwards every numbered title is pinned to that majority level."""
    from collections import Counter, defaultdict

    global_votes = defaultdict(Counter)
    for chunk in chunk_levels:
        for text, level in chunk:
            depth = _numbering_depth(text)
            if depth and level:
                global_votes[depth][level] += 1
    majority = {}
    for depth in sorted(global_votes):
        votes = global_votes[depth]
        if sum(votes.values()) < MIN_NUMBERING_VOTES:
            continue
        level = votes.most_common(1)[0][0]
        if majority and level <= max(majority.values()):
            continue  # deeper numbering must sit on a deeper level, otherwise the votes are noise
        majority[depth] = level
    if not majority:
        return

    for chunk_no, chunk in enumerate(chunk_levels, start=1):
        chunk_votes = defaultdict(Counter)
        for text, level in chunk:
            depth = _numbering_depth(text)
            if depth and level:
                chunk_votes[depth][level] += 1
        offsets = {
            votes.most_common(1)[0][0] - majority[depth]
            for depth, votes in chunk_votes.items()
            if depth in majority and sum(votes.values()) >= 2
        }
        if len(offsets) == 1 and (offset := offsets.pop()) != 0:
            logger.info(f"Chunk {chunk_no}: shifting all levels by {-offset} (numbering offset)")
            for i, (text, level) in enumerate(chunk):
                if level:
                    chunk[i] = (text, max(1, level - offset))
        for i, (text, level) in enumerate(chunk):
            depth = _numbering_depth(text)
            if depth and level and depth in majority:
                chunk[i] = (text, majority[depth])
            elif level and level > MAX_TITLE_LEVEL:
                chunk[i] = (text, MAX_TITLE_LEVEL)


def _run_chunked_title_leveling(title_block_refs, title_aided_config, chunk_size):
    context_size = int(title_aided_config.get("chunk_context", DEFAULT_CHUNK_CONTEXT))
    chunks = _split_title_chunks(title_block_refs, chunk_size)
    logger.info(
        f"LLM-aided title leveling in {len(chunks)} chunks "
        f"(chunk_size={chunk_size}, titles={len(title_block_refs)})"
    )
    leveled_entries = []
    chunk_levels = []
    for chunk_no, chunk in enumerate(chunks, start=1):
        title_dict = _build_title_dict(chunk)
        context = _build_chunk_context(leveled_entries, context_size)
        prompt_builder = None
        if context is not None:
            chunk_builder = _resolve_prompt_builder(title_aided_config, _build_chunk_title_optimize_prompt)
            prompt_builder = partial(chunk_builder, context)
        levels_by_index = _request_title_levels(title_aided_config, title_dict, prompt_builder=prompt_builder)
        if levels_by_index is None:
            logger.error(f"Chunk {chunk_no}/{len(chunks)} failed, its titles keep no level.")
            chunk_levels.append([(text, None) for text, _, _ in title_dict.values()])
            continue
        chunk_levels.append([(text, int(levels_by_index[int(key)])) for key, (text, _, _) in title_dict.items()])
        for key, (text, height, page) in title_dict.items():
            leveled_entries.append([text, height, page, int(levels_by_index[int(key)])])

    _realign_chunks_by_numbering(chunk_levels)
    for chunk, leveled in zip(chunks, chunk_levels):
        levels_by_index = {i: level for i, (_, level) in enumerate(leveled) if level is not None}
        if len(levels_by_index) == len(chunk):
            _apply_levels_to_blocks(chunk, levels_by_index)


def _split_paragraph_title_groups(title_block_refs):
    groups = []
    current_group = []

    for title_ref in title_block_refs:
        _, block = title_ref
        if block.get("type") == BlockType.DOC_TITLE:
            if current_group:
                groups.append(current_group)
                current_group = []
        elif block.get("type") == BlockType.PARAGRAPH_TITLE:
            current_group.append(title_ref)

    if current_group:
        groups.append(current_group)

    return groups


def _offset_paragraph_title_levels(levels_by_index):
    if not levels_by_index:
        return levels_by_index

    return {
        index: 2 if level == 1 else level
        for index, level in levels_by_index.items()
    }


def _request_paragraph_group_levels(title_block_refs, title_aided_config):
    title_dict = _build_title_dict(title_block_refs)
    levels_by_index = _request_title_levels(
        title_aided_config,
        title_dict,
        prompt_builder=_build_relative_title_optimize_prompt,
    )
    return _offset_paragraph_title_levels(levels_by_index)


def _run_grouped_title_leveling(title_block_refs, title_aided_config):
    doc_title_refs = []
    for title_ref in title_block_refs:
        _, block = title_ref
        if block.get("type") == BlockType.DOC_TITLE:
            block["level"] = 1
            doc_title_refs.append(title_ref)

    paragraph_title_groups = _split_paragraph_title_groups(title_block_refs)
    group_levels = []

    if len(paragraph_title_groups) > 1:
        max_workers = min(len(paragraph_title_groups), MAX_TITLE_GROUP_WORKERS)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _request_paragraph_group_levels,
                    title_group,
                    title_aided_config,
                )
                for title_group in paragraph_title_groups
            ]
            group_levels = [future.result() for future in futures]
    else:
        group_levels = [
            _request_paragraph_group_levels(title_group, title_aided_config)
            for title_group in paragraph_title_groups
        ]

    for title_group, levels_by_index in zip(paragraph_title_groups, group_levels):
        _apply_levels_to_blocks(title_group, levels_by_index)

    _normalize_title_types(doc_title_refs)
    for title_group in paragraph_title_groups:
        _normalize_title_types(title_group)


def llm_aided_title(page_info_list, title_aided_config):
    title_block_refs, title_types = _collect_title_block_refs(page_info_list)
    if len(title_block_refs) == 0:
        logger.info("No titles detected, skipping LLM-aided title optimization.")
        return

    has_doc_title = BlockType.DOC_TITLE in title_types
    has_paragraph_title = BlockType.PARAGRAPH_TITLE in title_types
    has_generic_title = BlockType.TITLE in title_types

    if has_doc_title and has_paragraph_title and not has_generic_title:
        _run_grouped_title_leveling(title_block_refs, title_aided_config)
        _sync_para_titles_to_preproc(page_info_list)
        return

    doc_title_refs = []
    title_refs_for_llm = []
    for title_ref in title_block_refs:
        _, block = title_ref
        if block.get("type") == BlockType.DOC_TITLE:
            block["level"] = 1
            doc_title_refs.append(title_ref)
        else:
            title_refs_for_llm.append(title_ref)

    chunk_size = int(title_aided_config.get("chunk_size", 0) or 0)
    if chunk_size > 0 and len(title_refs_for_llm) > chunk_size:
        _run_chunked_title_leveling(title_refs_for_llm, title_aided_config, chunk_size)
    elif len(title_refs_for_llm) > 0:
        _run_single_pass_title_leveling(title_refs_for_llm, title_aided_config)

    _normalize_title_types(doc_title_refs)
    _normalize_title_types(title_refs_for_llm)
    _sync_para_titles_to_preproc(page_info_list)
