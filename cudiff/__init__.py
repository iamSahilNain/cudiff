"""cudiff: differential testing of the CUDA compiler on integer kernels.

Each generated kernel body is a pure function of the thread index and three
input words, written once as a __host__ __device__ function. The same source
is compiled (a) by the host compiler as a plain CPU loop, the reference, and
(b) by nvcc at several device optimisation levels (-G, -Xptxas -O0..-O3), and
run on the GPU. Every binary prints a hash of the output array; any
disagreement between the GPU builds, or between a GPU build and the CPU
reference, is a candidate compiler bug. Kernel bodies are free of undefined
behaviour by construction (unsigned arithmetic, masked shifts, guarded
division, masked local-array indices), so a disagreement is never UB.
"""
__version__ = "0.1.0"
