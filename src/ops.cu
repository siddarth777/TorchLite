#include "ops.cuh"
#include <cuda_runtime.h>
#include <iostream>

#define CHECK_CUDA(x) do { \
  cudaError_t err = x; \
  if (err != cudaSuccess) { \
    std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    exit(EXIT_FAILURE); \
  } \
} while (0)

__global__ void add_kernel(const float* a, const float* b, float* out, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) out[idx] = a[idx] + b[idx];
}

__global__ void sub_kernel(const float* a, const float* b, float* out, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) out[idx] = a[idx] - b[idx];
}

__global__ void mul_kernel(const float* a, const float* b, float* out, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) out[idx] = a[idx] * b[idx];
}

__global__ void fill_kernel(float* data, float value, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = value;
}

__global__ void matmul_kernel(const float* a, const float* b, float* out, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += a[row * K + k] * b[k * N + col];
        }
        out[row * N + col] = sum;
    }
}

void launch_add(const float* a, const float* b, float* out, size_t size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>(a, b, out, size);
    CHECK_CUDA(cudaGetLastError());
}

void launch_sub(const float* a, const float* b, float* out, size_t size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    sub_kernel<<<blocks, threads>>>(a, b, out, size);
    CHECK_CUDA(cudaGetLastError());
}

void launch_mul(const float* a, const float* b, float* out, size_t size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    mul_kernel<<<blocks, threads>>>(a, b, out, size);
    CHECK_CUDA(cudaGetLastError());
}

void launch_fill(float* data, float value, size_t size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    fill_kernel<<<blocks, threads>>>(data, value, size);
    CHECK_CUDA(cudaGetLastError());
}

void launch_matmul(const float* a, const float* b, float* out, int M, int K, int N) {
    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);
    matmul_kernel<<<blocks, threads>>>(a, b, out, M, K, N);
    CHECK_CUDA(cudaGetLastError());
}
