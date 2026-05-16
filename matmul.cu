#include <iostream>
#include <cuda_runtime.h>

#define N 1024

// CUDA kernel
__global__
void matrixMulKernel(float* A, float* B, float* C, int n)
{
    // Row and column this thread computes
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds check
    if (row < n && col < n)
    {
        float sum = 0.0f;

        // Compute dot product
        for (int k = 0; k < n; k++)
        {
            sum += A[row * n + k] * B[k * n + col];
        }

        C[row * n + col] = sum;
    }
}

int main()
{
    size_t bytes = N * N * sizeof(float);

    // Host matrices
    float *h_A, *h_B, *h_C;

    h_A = (float*)malloc(bytes);
    h_B = (float*)malloc(bytes);
    h_C = (float*)malloc(bytes);

    // Initialize matrices
    for (int i = 0; i < N * N; i++)
    {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }

    // Device matrices
    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Copy data to GPU
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Thread block size
    dim3 threadsPerBlock(16, 16);

    // Grid size
    dim3 blocksPerGrid(
        (N + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (N + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    // Launch kernel
    matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Wait for GPU
    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // Example output
    std::cout << "C[0] = " << h_C[0] << std::endl;

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}