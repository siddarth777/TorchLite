#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
using namespace std;
#define N 1024

//CPU

void matrixMultiplyCPU(float* A, float* B, float* C, int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            float sum = 0.0f;

            for (int k = 0; k < n; k++)
            {
                sum += A[i * n + k] * B[k * n + j];
            }

            C[i * n + j] = sum;
        }
    }
}

// GPU 

__global__
void matrixMulKernel(float* A, float* B, float* C, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n)
    {
        float sum = 0.0f;

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

    float *A, *B, *C_cpu, *C_gpu;

    A = (float*)malloc(bytes);
    B = (float*)malloc(bytes);
    C_cpu = (float*)malloc(bytes);
    C_gpu = (float*)malloc(bytes);

    for (int i = 0; i < N * N; i++)
    {
        A[i] = 1.0f;
        B[i] = 1.0f;
    }

    

    auto cpu_start = chrono::high_resolution_clock::now();

    matrixMultiplyCPU(A, B, C_cpu, N);

    auto cpu_end = chrono::high_resolution_clock::now();

    chrono::duration<double> cpu_time = cpu_end - cpu_start;

    auto gpu_start = chrono::high_resolution_clock::now();

    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);

    dim3 blocksPerGrid(
        (N + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (N + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    // ================= GPU Timing =================

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaEventRecord(stop);

    cudaDeviceSynchronize();

    float gpu_ms = 0;

    cudaEventElapsedTime(&gpu_ms, start, stop);

    cudaMemcpy(C_gpu, d_C, bytes, cudaMemcpyDeviceToHost);

    auto gpu_end = chrono::high_resolution_clock::now();

    chrono::duration<double> gpu_time = gpu_end - gpu_start;

    cout << "CPU Time: " << cpu_time.count() << " seconds\n";

    cout << "GPU Time: " << gpu_ms / 1000.0f << " seconds\n";

    cout << "Speedup: "
              << cpu_time.count() / (gpu_ms / 1000.0f)
              << "x\n";

    cout << "CPU Result: " << C_cpu[0] << endl;
    cout << "GPU Result: " << C_gpu[0] << endl;

    cout << "GPU Totality Time: " << gpu_time.count() << " seconds\n";    


    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(A);
    free(B);
    free(C_cpu);
    free(C_gpu);

    return 0;
}