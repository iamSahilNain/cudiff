#include <cstdint>
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
__host__ __device__ static inline u32 compute(u32 i, i32 a, u32 b, i32 c) {
  u32 ua = (u32)a, uc = (u32)c; u32 arr[4] = {b, ua, uc, i};
  u32 v0 = (([&]() { u32 t = (clz(uc)); for (u32 k = 0; k < 2u; k++) t = (t * 2654435761u) ^ (k + (((255u) << ((arr[(b >> 3) & 3u]) & 31u)))); return t; }())); arr[(v0) & 3u] = v0 ^ arr[(i) & 3u];
  u32 v1 = (((((((2147483647u) + (255u))) & ((((uc) & 1u) ? (arr[(b >> 3) & 3u]) : (arr[(ua) & 3u]))))) >> ((((((arr[(i) & 3u]) / (((ua) & 1023u) + 1u))) >> ((v0) & 31u))) & 31u)));
  u32 v2 = (((u32)0u - (brev(arr[(ua) & 3u])))); arr[(v2) & 3u] = v2 ^ uc;
  u32 v3 = (clz(brev(((v2) - (b)))));
  u32 v4 = (([&]() { u32 t = (ua); for (u32 k = 0; k < 6u; k++) t = (t * 2654435761u) ^ (k + (brev(((u32)((v3) != (2147483648u)))))); return t; }()));
  u32 v5 = (((i) ^ (umulhi(umulhi(v4, arr[(ua) & 3u]), ((u32)(((i32)(arr[(ua) & 3u])) >> (int)((i) & 31u)))))));
  u32 v6 = (((u32)0u - (((umulhi(i, v1)) | (((v2) << ((v1) & 31u)))))));
  u32 v7 = (arr[(i) & 3u]);
  u32 v8 = (((((u32)s_min((i32)(arr[(i) & 3u]), (i32)(arr[(i) & 3u])))) + (((u32)(((i32)(ua)) >> (int)((7u) & 31u))))));
  u32 v9 = (v7);
  u32 v10 = (v1);
  u32 v11 = (((((((31u) & (v2))) % (((((u32)(((i32)(arr[(i) & 3u])) >> (int)((v9) & 31u)))) & 1023u) + 1u))) / (((((((u32)(((i32)(2u)) <= ((i32)(arr[(i) & 3u]))))) + (umulhi(arr[(i) & 3u], arr[(i) & 3u])))) & 1023u) + 1u)));
  u32 acc = 2166136261u;
  acc = (acc ^ v0) * 16777619u;
  acc = (acc ^ v1) * 16777619u;
  acc = (acc ^ v2) * 16777619u;
  acc = (acc ^ v3) * 16777619u;
  acc = (acc ^ v4) * 16777619u;
  acc = (acc ^ v5) * 16777619u;
  acc = (acc ^ v6) * 16777619u;
  acc = (acc ^ v7) * 16777619u;
  acc = (acc ^ v8) * 16777619u;
  acc = (acc ^ v9) * 16777619u;
  acc = (acc ^ v10) * 16777619u;
  acc = (acc ^ v11) * 16777619u;
  for (int k = 0; k < 4; k++) acc = (acc ^ arr[k]) * 16777619u;
  return acc;
}

#ifdef __CUDACC__
__global__ void kern(const i32* a, const u32* b, const i32* c, u32* out, u32 n) {
  u32 i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = compute(i, a[i], b[i], c[i]);
}
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("cuda error %s\n", cudaGetErrorString(e)); return 2; } } while (0)
#endif
static u64 fnv(const std::vector<u32>& v) { u64 h = 1469598103934665603ULL; for (u32 x : v) { h ^= x; h *= 1099511628211ULL; } return h; }
int main() {
  const u32 n = 4096;
  std::vector<i32> a(n); std::vector<u32> b(n); std::vector<i32> c(n); std::vector<u32> out(n, 0u);
  u64 s = 11400714819323198485ULL;
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
