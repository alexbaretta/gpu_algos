#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

// Matrix dimensions
constexpr int M = 1000;    // Rows of first matrix
constexpr int N = 10000;   // Columns of first matrix / Rows of second matrix
constexpr int K = 1000;    // Columns of second matrix

// CUDA kernel for matrix multiplication
__global__ void matrixMultiply(const float* A, const float* B, float* C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < k) {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            sum += A[row * n + i] * B[i * k + col];
        }
        C[row * k + col] = sum;
    }
}

// Function to initialize matrix with random values
void initializeMatrix(std::vector<float>& matrix, int rows, int cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = dis(gen);
    }
}

int main() {
    // Allocate host memory
    std::vector<float> h_A(M * N);
    std::vector<float> h_B(N * K);
    std::vector<float> h_C(M * K);

    // Initialize matrices
    initializeMatrix(h_A, M, N);
    initializeMatrix(h_B, N, K);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * N * sizeof(float));
    cudaMalloc(&d_B, N * K * sizeof(float));
    cudaMalloc(&d_C, M * K * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_A, h_A.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), N * K * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((K + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // Launch kernel
    matrixMultiply<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);

    // Wait for kernel to finish
    cudaDeviceSynchronize();

    // End timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    // Copy result back to host
    cudaMemcpy(h_C.data(), d_C, M * K * sizeof(float), cudaMemcpyDeviceToHost);

    // Print execution time
    std::cout << "Matrix multiplication completed in " << duration.count() << " seconds" << std::endl;

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
