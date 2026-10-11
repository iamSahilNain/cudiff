"""Shrink a divergent kernel by blanking statements (replace the expression with 0u)."""
from __future__ import annotations

import re
from typing import List

from .gen import Kernel
from .harness import check

_ASSIGN = re.compile(r"^u32 (v\d+) = \((.*)\);(.*)$")


def blank(stmt: str) -> str:
    m = _ASSIGN.match(stmt)
    if not m:
        return stmt
    return f"u32 {m.group(1)} = (0u);"


def reduce(kernel: Kernel, configs=None) -> Kernel:
    stmts: List[str] = list(kernel.stmts)
    changed = True
    while changed:
        changed = False
        for k in range(len(stmts)):
            if stmts[k] == blank(stmts[k]):
                continue
            trial = list(stmts)
            trial[k] = blank(stmts[k])
            if check(Kernel(kernel.seed, trial, kernel.n_threads), configs).kind == "diverge":
                stmts = trial
                changed = True
    return Kernel(kernel.seed, stmts, kernel.n_threads)
