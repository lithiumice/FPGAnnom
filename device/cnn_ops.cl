__kernel void conv(__global float *a, __global float *b, __global float *c,
                   const int M, const int N, const int K) {
  int gx = get_global_id(0);
  int gy = get_global_id(1);
  float tmp = 0.0f;
  for (int x = 0; x < K; x++) {
    for (int y = 0; y < K; y++) {
      tmp += a[(gx + x) * M + (gy + y)] * b[x * K + y];
    }
  }
}

__kernel void gemm(__global float *a, __global float *b, __global float *c,
                   const int M, const int N, const int K) {
  int gx = get_global_id(0);
  int gy = get_global_id(1);
  int sy = get_global_size(1);
  int sx = get_global_size(0);

  int s = sx * sy;
  for (int x = gx; x < M; x += sx) {
    for (int y = gy; y < N; y += sy) {
      float tmp = 0.0f;
      for (int z = 0; z < K; z++) {
        tmp += a[z * M + x] * b[y * K + z];
      }
      c[y * M + x] = tmp;
    }
  }
}