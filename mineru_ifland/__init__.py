"""Local extensions on top of MinerU 4.x (fork marifl/MinerU, branch local-4x).

Everything here builds on MinerU's public API and lives outside the `mineru` package,
so rebasing onto a new upstream release never conflicts with this code.
"""

# Bump on every behaviour change: callers put `mineru-de --version` into their cache fingerprint.
__version__ = "0.2.0"
