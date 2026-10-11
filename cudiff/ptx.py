"""Emit and compare the PTX that nvcc generates for one kernel at -G and optimised.

PTX is the virtual ISA between NVVM and ptxas; comparing the two builds shows
what the NVVM optimiser did to the kernel before ptxas ever ran.
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple

from .gen import Kernel
from .harness import gpu_arch, nvcc_path

_INSN = re.compile(r"^\s*([a-z][a-z0-9_.]*)\s", re.M)
_SKIP = {".reg", ".param", ".local", ".loc", ".file", ".version", ".target", ".address_size", ".visible", ".entry", ".func", ".shared", ".global", ".const", ".pragma", ".section", ".b8", ".b32", ".b64"}


def emit_ptx(kernel: Kernel, flags: List[str], wd: str, name: str) -> Optional[str]:
    nvcc = nvcc_path()
    if not nvcc:
        return None
    src = os.path.join(wd, "k.cu")
    if not os.path.exists(src):
        with open(src, "w") as f:
            f.write(kernel.source)
    out = os.path.join(wd, name + ".ptx")
    p = subprocess.run([nvcc, "-std=c++14", f"-arch={gpu_arch()}", "-w", "-x", "cu", "-ptx", *flags, src, "-o", out],
                       capture_output=True, text=True, timeout=120)
    if p.returncode != 0:
        return None
    return out


def summarize(ptx_path: str) -> Dict[str, int]:
    text = open(ptx_path).read()
    # count every instruction in the file: at -G the compute() body is a separate
    # non-inlined .func placed before the .entry, so restricting to the entry would
    # hide most of the unoptimised code.
    body = text
    counts: Dict[str, int] = {}
    for m in _INSN.finditer(body):
        op = m.group(1)
        if op.startswith(".") or op.startswith("//") or op.endswith(":"):
            continue
        base = op.split(".")[0]
        counts[base] = counts.get(base, 0) + 1
    counts["_total"] = sum(v for k, v in counts.items() if not k.startswith("_"))
    counts["_lines"] = len(body.splitlines())
    return counts


def compare(kernel: Kernel, keep_dir: Optional[str] = None) -> Tuple[Dict[str, int], Dict[str, int], str, str]:
    wd = keep_dir or tempfile.mkdtemp(prefix="cudiff_ptx_")
    os.makedirs(wd, exist_ok=True)
    g = emit_ptx(kernel, ["-G"], wd, "kernel_G")
    o = emit_ptx(kernel, [], wd, "kernel_opt")
    if g is None or o is None:
        raise RuntimeError("nvcc -ptx failed")
    return summarize(g), summarize(o), g, o
