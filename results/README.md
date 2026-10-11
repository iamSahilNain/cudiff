# Campaigns

Filled in from `summary.json` after each run; nothing is estimated.

| Campaign | Kernels | Statements per kernel | Threads per kernel | Configs | agree | diverge | compiler-crash | harness-error |
|---|---|---|---|---|---|---|---|---|
| c1_std (11 Oct 2026, 2x RTX A4000 sm_86, CUDA 11.8, g++ 9.4) | 1000 | 12 | 4096 | 7 (cpu-O0, cpu-O2, nvcc-G, nvcc-ptxas-O0, nvcc-ptxas-O1, nvcc-ptxas-O2, nvcc-ptxas-O3) | 1000 | 0 | 0 | 0 |
| c2_big (11 Oct 2026, 2x RTX A4000 sm_86, CUDA 11.8, g++ 9.4) | 300 | 24 | 4096 | 7 (cpu-O0, cpu-O2, nvcc-G, nvcc-ptxas-O0, nvcc-ptxas-O1, nvcc-ptxas-O2, nvcc-ptxas-O3) | 300 | 0 | 0 | 0 |

Totals: 1,300 kernels, 7 builds each (2 CPU, 5 GPU), 5.3 million thread executions per config, 0 divergences, 0 compiler crashes.
