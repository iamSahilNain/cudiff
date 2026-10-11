"""Random integer-kernel generator (UB-free by construction, Python 3.8 compatible)."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List

TEMPLATE_HEAD = r'''#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
#define __host__
#define __device__
#endif
typedef uint32_t u32; typedef int32_t i32; typedef uint64_t u64; typedef int64_t i64;
__host__ __device__ static inline u32 popc(u32 x) {
#ifdef __CUDA_ARCH__
  return (u32)__popc((int)x);
#else
  return (u32)__builtin_popcount(x);
#endif
}
__host__ __device__ static inline u32 clz(u32 x) {
  if (x == 0u) return 32u;
#ifdef __CUDA_ARCH__
  return (u32)__clz((int)x);
#else
  return (u32)__builtin_clz(x);
#endif
}
__host__ __device__ static inline u32 brev(u32 x) {
#ifdef __CUDA_ARCH__
  return __brev(x);
#else
  u32 r = 0; for (int k = 0; k < 32; k++) { r = (r << 1) | (x & 1u); x >>= 1; } return r;
#endif
}
__host__ __device__ static inline u32 umulhi(u32 a, u32 b) {
#ifdef __CUDA_ARCH__
  return __umulhi(a, b);
#else
  return (u32)(((u64)a * (u64)b) >> 32);
#endif
}
__host__ __device__ static inline u32 u_min(u32 a, u32 b) { return a < b ? a : b; }
__host__ __device__ static inline u32 u_max(u32 a, u32 b) { return a > b ? a : b; }
__host__ __device__ static inline i32 s_min(i32 a, i32 b) { return a < b ? a : b; }
__host__ __device__ static inline i32 s_max(i32 a, i32 b) { return a > b ? a : b; }
'''

TEMPLATE_TAIL = r'''
#ifdef __CUDACC__
__global__ void kern(const i32* a, const u32* b, const i32* c, u32* out, u32 n) {
  u32 i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = compute(i, a[i], b[i], c[i]);
}
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("cuda error %s\n", cudaGetErrorString(e)); return 2; } } while (0)
#endif
static u64 fnv(const std::vector<u32>& v) { u64 h = 1469598103934665603ULL; for (u32 x : v) { h ^= x; h *= 1099511628211ULL; } return h; }
int main() {
  const u32 n = N_THREADS;
  std::vector<i32> a(n); std::vector<u32> b(n); std::vector<i32> c(n); std::vector<u32> out(n, 0u);
  u64 s = SEED_LCG;
  for (u32 i = 0; i < n; i++) {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL; a[i] = (i32)(u32)(s >> 32);
    s = s * 6364136223846793005ULL + 1442695040888963407ULL; b[i] = (u32)(s >> 32);
    s = s * 6364136223846793005ULL + 1442695040888963407ULL; c[i] = (i32)(u32)(s >> 32) % 4096;
  }
  for (u32 i = 0; i < n; i++) out[i] = compute(i, a[i], b[i], c[i]);
  printf("cpu=%016llx\n", (unsigned long long)fnv(out));
#ifdef __CUDACC__
  i32 *da; u32 *db; i32 *dc; u32 *dout;
  CK(cudaMalloc(&da, n * 4)); CK(cudaMalloc(&db, n * 4)); CK(cudaMalloc(&dc, n * 4)); CK(cudaMalloc(&dout, n * 4));
  CK(cudaMemcpy(da, a.data(), n * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(db, b.data(), n * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dc, c.data(), n * 4, cudaMemcpyHostToDevice)); CK(cudaMemset(dout, 0, n * 4));
  kern<<<(n + 255) / 256, 256>>>(da, db, dc, dout, n);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  std::vector<u32> g(n); CK(cudaMemcpy(g.data(), dout, n * 4, cudaMemcpyDeviceToHost));
  printf("gpu=%016llx\n", (unsigned long long)fnv(g));
  u32 first = n; for (u32 i = 0; i < n; i++) if (g[i] != out[i]) { first = i; break; }
  if (first < n) printf("first_diff=%u cpu=%08x gpu=%08x a=%08x b=%08x c=%08x\n", first, out[first], g[first], (u32)a[first], b[first], (u32)c[first]);
#endif
  return 0;
}
'''

U_OPS = ["+", "-", "*", "&", "|", "^"]


@dataclass
class Kernel:
    seed: int
    stmts: List[str]           # "vK = <expr>;" statements, in order
    n_threads: int = 4096

    @property
    def source(self) -> str:
        body = "\n".join("  " + s for s in self.stmts)
        fn = ("__host__ __device__ static inline u32 compute(u32 i, i32 a, u32 b, i32 c) {\n"
              "  u32 ua = (u32)a, uc = (u32)c; u32 arr[4] = {b, ua, uc, i};\n"
              f"{body}\n"
              "  u32 acc = 2166136261u;\n" +
              "".join(f"  acc = (acc ^ v{k}) * 16777619u;\n" for k in range(len(self.stmts))) +
              "  for (int k = 0; k < 4; k++) acc = (acc ^ arr[k]) * 16777619u;\n"
              "  return acc;\n}\n")
        return (TEMPLATE_HEAD + fn + TEMPLATE_TAIL.replace("N_THREADS", str(self.n_threads))
                .replace("SEED_LCG", f"{(self.seed * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF}ULL"))


class Gen:
    def __init__(self, seed: int, n_stmts: int = 12, depth: int = 3):
        self.rng = random.Random(seed)
        self.seed = seed
        self.n_stmts = n_stmts
        self.depth = depth
        self.vars: List[str] = []

    def leaf(self) -> str:
        r = self.rng.random()
        if r < 0.15:
            return f"{self.rng.choice([0, 1, 2, 3, 7, 31, 255, 65535, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, self.rng.getrandbits(32)])}u"
        pool = ["i", "ua", "b", "uc"] + self.vars + ["arr[(ua) & 3u]", "arr[(i) & 3u]", "arr[(b >> 3) & 3u]"]
        return self.rng.choice(pool)

    def expr(self, d: int) -> str:
        if d <= 0 or self.rng.random() < 0.2:
            return self.leaf()
        op = self.rng.choice(U_OPS + ["<<", ">>", "sra", "/", "%", "cmp", "scmp", "?:", "popc", "clz", "brev",
                                      "umulhi", "u_min", "u_max", "s_min", "s_max", "~", "neg", "loop"])
        e = lambda: self.expr(d - 1)
        if op in U_OPS:
            return f"(({e()}) {op} ({e()}))"
        if op == "<<":
            return f"(({e()}) << (({e()}) & 31u))"
        if op == ">>":
            return f"(({e()}) >> (({e()}) & 31u))"
        if op == "sra":
            return f"((u32)(((i32)({e()})) >> (int)(({e()}) & 31u)))"
        if op in ("/", "%"):
            return f"(({e()}) {op} ((({e()}) & 1023u) + 1u))"
        if op == "cmp":
            c = self.rng.choice(["<", "<=", "==", "!=", ">", ">="])
            return f"((u32)(({e()}) {c} ({e()})))"
        if op == "scmp":
            c = self.rng.choice(["<", "<=", ">", ">="])
            return f"((u32)(((i32)({e()})) {c} ((i32)({e()}))))"
        if op == "?:":
            return f"((({e()}) & 1u) ? ({e()}) : ({e()}))"
        if op in ("popc", "clz", "brev"):
            return f"{op}({e()})"
        if op in ("umulhi", "u_min", "u_max"):
            return f"{op}({e()}, {e()})"
        if op in ("s_min", "s_max"):
            return f"((u32){op}((i32)({e()}), (i32)({e()})))"
        if op == "~":
            return f"(~({e()}))"
        if op == "neg":
            return f"((u32)0u - ({e()}))"
        if op == "loop":
            trips = self.rng.randint(1, 8)
            return f"([&]() {{ u32 t = ({e()}); for (u32 k = 0; k < {trips}u; k++) t = (t * 2654435761u) ^ (k + ({e()})); return t; }}())"
        return self.leaf()

    def generate(self) -> Kernel:
        stmts = []
        for k in range(self.n_stmts):
            r = self.rng.random()
            if r < 0.15:
                stmts.append(f"u32 v{k} = ({self.expr(self.depth - 1)}); arr[(v{k}) & 3u] = v{k} ^ {self.leaf()};")
            else:
                stmts.append(f"u32 v{k} = ({self.expr(self.depth)});")
            self.vars.append(f"v{k}")
        return Kernel(self.seed, stmts)


def kernel(seed: int, n_stmts: int = 12, depth: int = 3) -> Kernel:
    return Gen(seed, n_stmts, depth).generate()
