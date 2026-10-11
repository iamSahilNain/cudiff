"""Compile one kernel with every config, run, compare hashes (Python 3.8 compatible)."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .gen import Kernel


@dataclass(frozen=True)
class Config:
    name: str
    cmd: List[str]          # compiler command prefix; source and -o appended
    gpu: bool


def host_compiler() -> Optional[str]:
    for c in (os.environ.get("CXX"), "g++", "clang++"):
        if c and shutil.which(c):
            return shutil.which(c)
    return None


def nvcc_path() -> Optional[str]:
    for c in (os.environ.get("NVCC"), "nvcc", "/usr/local/cuda/bin/nvcc"):
        if c and (shutil.which(c) or os.path.exists(c)):
            return shutil.which(c) or c
    return None


def gpu_arch() -> str:
    env = os.environ.get("CUDIFF_ARCH")
    if env:
        return env
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
        if out:
            return "sm_" + out[0].strip().replace(".", "")
    except (OSError, subprocess.TimeoutExpired):
        pass
    return "sm_86"


def default_configs() -> List[Config]:
    cfgs: List[Config] = []
    cxx = host_compiler()
    if cxx:
        cfgs.append(Config("cpu-O0", [cxx, "-std=c++14", "-O0", "-w", "-x", "c++"], False))
        cfgs.append(Config("cpu-O2", [cxx, "-std=c++14", "-O2", "-w", "-x", "c++"], False))
    nvcc = nvcc_path()
    if nvcc:
        arch = gpu_arch()
        base = [nvcc, "-std=c++14", f"-arch={arch}", "-w", "-x", "cu"]
        cfgs.append(Config("nvcc-G", base + ["-G"], True))
        for o in ("0", "1", "2", "3"):
            cfgs.append(Config(f"nvcc-ptxas-O{o}", base + ["-Xptxas", f"-O{o}"], True))
    return cfgs


@dataclass
class RunResult:
    config: str
    status: str                       # ok | compile-error | compile-timeout | run-error | run-timeout
    cpu: Optional[str] = None
    gpu: Optional[str] = None
    detail: str = ""


@dataclass
class Verdict:
    kind: str                         # agree | diverge | compiler-crash | harness-error
    results: List[RunResult] = field(default_factory=list)
    groups: Dict[str, List[str]] = field(default_factory=dict)
    note: str = ""


_HASH = re.compile(r"^(cpu|gpu)=([0-9a-f]{16})$", re.M)


def compile_and_run(cfg: Config, src: str, wd: str, compile_timeout: float = 120.0, run_timeout: float = 60.0) -> RunResult:
    exe = os.path.join(wd, "bin_" + cfg.name)
    try:
        p = subprocess.run(cfg.cmd + [src, "-o", exe], capture_output=True, text=True, timeout=compile_timeout, cwd=wd)
    except subprocess.TimeoutExpired:
        return RunResult(cfg.name, "compile-timeout")
    if p.returncode != 0:
        return RunResult(cfg.name, "compile-error", detail=(p.stderr or p.stdout)[-1500:])
    try:
        r = subprocess.run([exe], capture_output=True, text=True, timeout=run_timeout, cwd=wd)
    except subprocess.TimeoutExpired:
        return RunResult(cfg.name, "run-timeout")
    if r.returncode != 0:
        return RunResult(cfg.name, "run-error", detail=(r.stdout + r.stderr)[-800:])
    found = dict(_HASH.findall(r.stdout))
    res = RunResult(cfg.name, "ok", cpu=found.get("cpu"), gpu=found.get("gpu"), detail=r.stdout.strip()[-300:])
    if res.cpu is None or (cfg.gpu and res.gpu is None):
        res.status = "run-error"
        res.detail = "missing hash line: " + r.stdout[-300:]
    return res


def check(kernel: Kernel, configs: Optional[List[Config]] = None, keep_dir: Optional[str] = None) -> Verdict:
    configs = configs or default_configs()
    wd = keep_dir or tempfile.mkdtemp(prefix="cudiff_")
    try:
        src = os.path.join(wd, "k.cu")
        with open(src, "w") as f:
            f.write(kernel.source)
        results = [compile_and_run(c, src, wd) for c in configs]
    finally:
        if keep_dir is None:
            shutil.rmtree(wd, ignore_errors=True)
    v = Verdict("agree", results=results)
    bad = [r for r in results if r.status in ("compile-error", "compile-timeout")]
    if bad:
        v.kind = "compiler-crash"
        v.note = "; ".join(f"{r.config}: {r.status} {r.detail[:200]}" for r in bad)
        return v
    errs = [r for r in results if r.status != "ok"]
    if errs:
        v.kind = "harness-error"
        v.note = "; ".join(f"{r.config}: {r.status} {r.detail[:200]}" for r in errs)
        return v
    # every observation: (config, cpu or gpu, hash)
    groups: Dict[str, List[str]] = {}
    for r in results:
        groups.setdefault(r.cpu, []).append(r.config + "/cpu")
        if r.gpu is not None:
            groups.setdefault(r.gpu, []).append(r.config + "/gpu")
    v.groups = groups
    if len(groups) > 1:
        v.kind = "diverge"
        v.note = " vs ".join(",".join(g) for g in groups.values())
        for r in results:
            if "first_diff" in r.detail:
                v.note += " | " + r.detail.splitlines()[-1]
                break
    return v
