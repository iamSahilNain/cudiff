# Interview notes for cudiff

**One sentence.** cudiff generates integer CUDA kernels that cannot have undefined behaviour, compiles each one as a CPU loop and with nvcc at five device optimisation levels, runs all of them, and reports any output disagreement as a candidate compiler bug.

**Why one source for both targets.** The kernel body is a `__host__ __device__` function of the thread index and three input words. The CPU reference is the same function compiled by g++; it cannot drift from the kernel, and it costs nothing to write. The GPU wrapper just maps thread index to array index.

**Why UB-free by construction matters.** If the kernel had undefined behaviour the compilers would be allowed to disagree. So: unsigned arithmetic where overflow is possible, shift counts masked to 31, divisors forced into 1..1024, local-array indices masked, signed values only in comparisons, min/max and arithmetic right shift. Then a disagreement is a bug in g++ or in nvcc, never in the program.

**What the configs separate.** `-G` turns off optimisation in NVVM (the LLVM-based front half that emits PTX) and in ptxas (the PTX-to-SASS assembler). `-Xptxas -O0` to `-O3` vary ptxas only, with NVVM at its default optimisation. A disagreement between `-G` and `-Xptxas -O0` points at NVVM; between two ptxas levels, at ptxas; between the CPU and every GPU build, at the front end or at a semantic difference such as `__clz(0)`.

**GPU-specific pieces.** `__popc`, `__clz`, `__brev`, `__umulhi` with CPU equivalents (`__clz(0)` is 32 on the device; `__builtin_clz(0)` is undefined on the host, so the wrapper guards zero). A four-word local array indexed by data exercises local memory. Bounded loops with data-dependent bodies exercise the optimiser.

**How I know the detector detects.** `tests/test_harness.py` wraps g++ in a script that rewrites the hash multiplier before compiling; the harness must report a divergence. A compiler that always exits 1 must be reported as a crash. Generated kernels must compile and agree on the host. These run without a GPU.

**What a candidate becomes.** The reducer blanks one statement at a time (replaces its expression with `0u`) while the disagreement persists, giving a minimal kernel. The candidate's source, the per-config hashes and the first differing thread with its inputs are saved.

**Numbers.** Only what `results/README.md` says. The campaigns ran on two RTX A4000 (sm_86) with CUDA 11.8 on a lab machine.

**Honest limits.** Integer only; no floats, shared memory, warp intrinsics or atomics; threads do not communicate; the CPU reference is itself a compiler; a few thousand kernels against a release toolkit rarely find a bug. I wrote the harness in a day and had not written CUDA before it, so what I can defend is the method, the UB argument, the NVVM-versus-ptxas split and the generator, not CUDA performance work.

**Likely follow-ups.** What PTX is and where it sits (virtual ISA between NVVM and SASS); why `-G` changes codegen; what a miscompile looks like versus a crash; why hashing the output is enough (and when it is not: order-dependent reductions); how to extend to shared memory (per-block reference), to floats (tolerances, FMA contraction, `--fmad=false`), and to PTX-level checks (`-ptx` output diffs, `cuobjdump -sass`).
