#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cxxopts.hpp>

#include "common/cuda/check_errors.h"
#include "common/cuda/cuda_utils.h"

// Default matrix dimensions
constexpr int DEFAULT_M = 1000;    // Rows of first matrix
constexpr int DEFAULT_N = 10000;   // Columns of first matrix / Rows of second matrix
constexpr unsigned int NULL_FLAGS = 0;

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
void initialize_matrix(std::vector<float>& matrix, int rows, int cols) {
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

        std::cout << "CPU:" << std::endl;
        auto t0 = std::chrono::high_resolution_clock::now();

        std::cout << "  - Allocating memory: ";
        std::vector<float> h_A(size, 0.0f);
        std::vector<float> h_B(size, 0.0f);
        std::vector<float> h_C(size, 0.0f);
        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> dt1 = t1 - t0;
        std::cout << dt1.count() << " s (" << dt1.count() << " s total)" << std::endl;

        std::cout << "  - Initializing matrices: ";
        initialize_matrix(h_A, M, N);
        initialize_matrix(h_B, N, M);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> dt2 = t2 - t1;
        std::cout << dt2.count() << " s (" << dt2.count() << " s total)" << std::endl;

        std::cout << "  - Creating GPU streams: ";
        cudaStream_t stream;
        cuda_check_error(cudaStreamCreate(&stream), "cudaStreamCreate");
        auto t3 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> dt3 = t3 - t2;
        std::cout << dt3.count() << " s (" << dt3.count() << " s total)" << std::endl;

        std::cout << "  - Creating GPU events: ";
        cudaEvent_t e0, e1, e2, e3, e4, e5;
        cuda_check_error(cudaEventCreate(&e0), "cudaEventCreate");
        cuda_check_error(cudaEventCreate(&e1), "cudaEventCreate");
        cuda_check_error(cudaEventCreate(&e2), "cudaEventCreate");
        cuda_check_error(cudaEventCreate(&e3), "cudaEventCreate");
        cuda_check_error(cudaEventCreate(&e4), "cudaEventCreate");
        cuda_check_error(cudaEventCreate(&e5), "cudaEventCreate");
        auto t4 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> dt4 = t4 - t3;
        std::cout << dt4.count() << " s (" << dt4.count() << " s total)" << std::endl;


        std::cout << "GPU:" << std::endl;
        cuda_check_error(cudaEventRecord(e0, stream), "cudaEventRecord");

        float *d_A, *d_B, *d_C;
        cudaMallocAsync(&d_A, size * sizeof(float), stream);
        cudaMallocAsync(&d_B, size * sizeof(float), stream);
        cudaMallocAsync(&d_C, size * sizeof(float), stream);
        cuda_check_error(cudaEventRecord(e1, stream), "cudaEventRecord");
        auto step1 = std::make_tuple(e1, e0, e0, "allocating device memory");
        cudaStreamAddCallback(stream, report_completion_callback, &step1, NULL_FLAGS);

        // Copy data to device
        cudaMemcpy(d_A, h_A.data(), size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B.data(), size * sizeof(float), cudaMemcpyHostToDevice);
        cuda_check_error(cudaEventRecord(e1, stream), "cudaEventRecord");
        auto step2 = std::make_tuple(e2, e1, e0, "copying data to device");
        cudaStreamAddCallback(stream, report_completion_callback, &step2, NULL_FLAGS);

        // Define block and grid dimensions
        dim3 blockDim(16, 16);
        dim3 gridDim((M + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

        // Compute kernel
        matrixMultiply<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, M);
        cuda_check_error(cudaEventRecord(e3, stream), "cudaEventRecord");
        auto step3 = std::make_tuple(e3, e2, e0, "computing kernel");
        cudaStreamAddCallback(stream, report_completion_callback, &step3, NULL_FLAGS);

        // Copy result back to host
        cudaMemcpyAsync(h_C.data(), d_C, size * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cuda_check_error(cudaEventRecord(e4, stream), "cudaEventRecord");
        auto step4 = std::make_tuple(e4, e3, e0, "copying result to host");
        cudaStreamAddCallback(stream, report_completion_callback, &step4, NULL_FLAGS);


        // Free device memory
        cudaFreeAsync(d_A, stream);
        cudaFreeAsync(d_B, stream);
        cudaFreeAsync(d_C, stream);
        cuda_check_error(cudaEventRecord(e5, stream), "cudaEventRecord");
        auto step5 = std::make_tuple(e5, e4, e0, "freeing device memory");
        cudaStreamAddCallback(stream, report_completion_callback, &step5, NULL_FLAGS);

        // Wait for stream to finish
        cudaStreamSynchronize(stream);


        // Print execution time
        auto t5 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> dt5 = t5 - t4;
        std::cout << "DONE: " << dt5.count() << " s (" << dt5.count() << " s total)" << std::endl;


    } catch (const cxxopts::exceptions::exception& e) {
        std::cerr << "Error parsing options: " << e.what() << std::endl;
        std::cout << options.help() << std::endl;
        return 1;
    }

    return 0;
}
