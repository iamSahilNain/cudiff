# cudiff — differential testing of the CUDA compiler on integer kernels

Generate a random, undefined-behaviour-free integer kernel; compile the same
source as a CPU loop (the reference) and with `nvcc` at several device
optimisation levels; run everything; compare a hash of the output array. Any
disagreement between two GPU builds, or between a GPU build and the CPU
reference, is a candidate compiler bug. Divergent kernels are shrunk
statement by statement until the disagreement disappears.

```
generator ──> k.cu ──> g++ -O0 / g++ -O2            (CPU reference, same compute())
                   ├─> nvcc -G                       (NVVM and ptxas unoptimised)
                   ├─> nvcc -Xptxas -O0 … -O3        (NVVM optimised, ptxas varied)
                   └─> run each, hash out[], compare ──> agree | diverge | compiler-crash
```

## Why this shape

* **One source, two targets.** The kernel body is a `__host__ __device__`
  function of `(threadIdx, a[i], b[i], c[i])`. The CPU reference is the same
  code compiled by the host compiler, so the oracle costs nothing to write
  and cannot drift from the kernel.
* **UB-free by construction.** Overflow-prone arithmetic is unsigned, shift
  counts are masked to 31, divisors are forced into 1..1024, local-array
  indices are masked to the array length, and signed operations are limited
  to comparisons, `min`/`max` and arithmetic right shift. A disagreement is
  therefore never "the program was undefined".
* **GPU-specific operations.** `__popc`, `__clz`, `__brev`, `__umulhi`,
  bounded loops and a per-thread local array exercise the parts of the
  device code path that a plain C test does not.
* **Both halves of the device compiler.** `-G` turns off optimisation in
  NVVM and ptxas; `-Xptxas -O0..-O3` varies the PTX-to-SASS assembler while
  NVVM stays optimised. A divergence between `-G` and `-Xptxas -O0` points at
  NVVM; between ptxas levels, at ptxas.

## Usage

```
python -m cudiff configs                       # compilers found (nvcc optional)
python -m cudiff gen --seed 7                  # print one kernel
python -m cudiff run -n 500 --jobs 8 --reduce  # campaign, shrink any failures
pytest -q                                      # harness tests (no GPU needed)
```

Without `nvcc` the harness still runs the CPU configs, which is what the
tests use: `tests/test_harness.py` proves that a planted miscompile (a
wrapped compiler that rewrites a constant before compiling) is reported as a
divergence, that a compiler that always exits 1 is reported as a crash, and
that generated kernels compile and agree on the host.

Python 3.8 or newer, no dependencies; nvcc 11.8 and an Ampere GPU were used
for the campaigns.

## Results

See `results/README.md`.

## Limitations

* Integer kernels only; no floating point, no shared memory, no warp
  intrinsics, no atomics, no inter-thread communication.
* The CPU reference is a compiler too; if g++ and nvcc agree on a wrong
  answer the harness cannot see it.
* Finding a bug in a release CUDA toolkit with a few thousand kernels is
  unlikely; the harness and its self-verification are the deliverable, and
  campaign counts are reported as they come out.
