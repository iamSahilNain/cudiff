"""Detection is proven against planted bugs; generator validity is checked by compiling."""
import os
import stat
import tempfile

import pytest

from cudiff.gen import kernel
from cudiff.harness import Config, check, host_compiler

CXX = host_compiler()
pytestmark = pytest.mark.skipif(CXX is None, reason="no host C++ compiler")


def _script(path, body):
    with open(path, "w") as f:
        f.write("#!/bin/bash\n" + body)
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)
    return path


@pytest.mark.parametrize("seed", range(1, 16))
def test_generated_kernels_compile_and_agree_on_host(seed):
    cfgs = [Config("cpu-O0", [CXX, "-std=c++14", "-O0", "-w", "-x", "c++"], False),
            Config("cpu-O2", [CXX, "-std=c++14", "-O2", "-w", "-x", "c++"], False)]
    v = check(kernel(seed), cfgs)
    assert v.kind == "agree", (v.kind, v.note)


def test_planted_miscompile_is_reported():
    with tempfile.TemporaryDirectory() as wd:
        # a "compiler" that rewrites the hash multiplier 16777619u to 16777620u before compiling;
        # argument layout from the harness is: <prefix flags...> <src> -o <exe>  (bash 3.2 compatible)
        bad = _script(os.path.join(wd, "badcxx"),
                      f'n=$#; exe="${{!n}}"; m=$((n-2)); src="${{!m}}"; '
                      f'sed "s/16777619u/16777620u/" "$src" > "$src.bad.cpp"; '
                      f'exec {CXX} "${{@:1:$((n-3))}}" "$src.bad.cpp" -o "$exe"\n')
        cfgs = [Config("cpu-O0", [CXX, "-std=c++14", "-O0", "-w", "-x", "c++"], False),
                Config("bad-O0", [bad, "-std=c++14", "-O0", "-w", "-x", "c++"], False)]
        v = check(kernel(3), cfgs)
        assert v.kind == "diverge", (v.kind, v.note)
        assert len(v.groups) == 2


def test_compiler_crash_is_reported():
    with tempfile.TemporaryDirectory() as wd:
        crash = _script(os.path.join(wd, "crashcxx"), "echo 'internal compiler error' >&2; exit 1\n")
        cfgs = [Config("cpu-O0", [CXX, "-std=c++14", "-O0", "-w", "-x", "c++"], False),
                Config("crash", [crash], False)]
        v = check(kernel(3), cfgs)
        assert v.kind == "compiler-crash" and "crash" in v.note
