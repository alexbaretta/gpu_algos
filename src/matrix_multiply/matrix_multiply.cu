#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cxxopts.hpp>

// Default matrix dimensions
constexpr int DEFAULT_M = 1000;    // Rows of first matrix
constexpr int DEFAULT_N = 10000;   // Columns of first matrix / Rows of second matrix
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

int main(int argc, char** argv) {
    // Parse command line arguments
    cxxopts::Options options("matrix_multiply", "CUDA Matrix Multiplication");
    options.add_options()
        ("nrows", "Number of rows in first matrix", cxxopts::value<int>()->default_value(std::to_string(DEFAULT_M)))
        ("ncols", "Number of columns in first matrix", cxxopts::value<int>()->default_value(std::to_string(DEFAULT_N)))
        ("h,help", "Print usage");

    try {
        auto result = options.parse(argc, argv);

        if (result.count("help")) {
            std::cout << options.help() << std::endl;
            return 0;
        }

        // Get matrix dimensions from command line or use defaults
        const int M = result["nrows"].as<int>();
        const int N = result["ncols"].as<int>();
        const int size = M * N;

        std::cout << "Matrix dimensions: " << M << "x" << N << " * " << N << "x" << M << std::endl;

        // Allocate host memory
        std::vector<float> h_A(size);
        std::vector<float> h_B(size);
        std::vector<float> h_C(size);

        // Initialize matrices
        initializeMatrix(h_A, M, N);
        initializeMatrix(h_B, N, M);

        // Allocate device memory
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, size * sizeof(float));
        cudaMalloc(&d_B, size * sizeof(float));
        cudaMalloc(&d_C, size * sizeof(float));

        // Copy data to device
        cudaMemcpy(d_A, h_A.data(), size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B.data(), size * sizeof(float), cudaMemcpyHostToDevice);

        // Define block and grid dimensions
        dim3 blockDim(16, 16);
        dim3 gridDim((M + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

        // Start timing
        auto start = std::chrono::high_resolution_clock::now();

        // Launch kernel
        matrixMultiply<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, M);

        // Wait for kernel to finish
        cudaDeviceSynchronize();

        // End timing
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;

        // Copy result back to host
        cudaMemcpy(h_C.data(), d_C, size * sizeof(float), cudaMemcpyDeviceToHost);

        // Print execution time
        std::cout << "Matrix multiplication completed in " << duration.count() << " seconds" << std::endl;

        // Free device memory
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

    } catch (const cxxopts::exceptions::exception& e) {
        std::cerr << "Error parsing options: " << e.what() << std::endl;
        std::cout << options.help() << std::endl;
        return 1;
    }

    return 0;
}
