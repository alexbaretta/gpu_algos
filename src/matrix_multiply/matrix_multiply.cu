// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

#include <chrono>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <random>
#include <ratio>
#include <vector>

#include <cuda_runtime.h>
#include <cxxopts.hpp>
#include <Eigen/Core>

#include "common/cuda/check_errors.h"
#include "common/cuda/cuda_utils.h"
#include "common/eigen/algorithms.h"
#include "common/random.h"
#ifndef _OPENMP
static_assert(false, "OpenMP is not supported");
#endif

// Default matrix dimensions
constexpr int DEFAULT_M = 3000;     // Rows of first matrix
constexpr int DEFAULT_N = 300;      // Columns of first matrix / Rows of second matrix
constexpr int DEFAULT_GPU_MEM = 16; // GPU memory size in GB
constexpr unsigned int NULL_FLAGS = 0;
constexpr int DEFAULT_SEED = 42;
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

int main(int argc, char** argv) {
    // Parse command line arguments
    cxxopts::Options options("matrix_multiply", "CUDA Matrix Multiplication");
    options.add_options()
        ("nrows", "Number of rows in first matrix", cxxopts::value<int>()->default_value(std::to_string(DEFAULT_M)))
        ("ncols", "Number of columns in first matrix", cxxopts::value<int>()->default_value(std::to_string(DEFAULT_N)))
        ("gpumem", "GPU memory size", cxxopts::value<int>()->default_value(std::to_string(DEFAULT_GPU_MEM)))
        ("seed", "Random seed", cxxopts::value<int>()->default_value(std::to_string(DEFAULT_SEED)))
        ("verbose", "Verbose output", cxxopts::value<bool>()->default_value("false"))
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
        const int seed = result["seed"].as<int>();
        const int input_size = M * N;
        const int output_size = M * M;
        const int gpu_mem = result["gpumem"].as<int>();
        const bool verbose = result["verbose"].as<bool>();
        constexpr float GB = 1024.0f * 1024.0f * 1024.0f;
        const float input_matrix_size_gb = input_size * sizeof(float) / GB;
        const float output_matrix_size_gb = output_size * sizeof(float) / GB;
        const float mem_gb = 2*input_matrix_size_gb + output_matrix_size_gb;
        std::cout
            << "Matrix dimensions   : " << M << "x" << N << " * " << N << "x" << M << "\n"
            << "Input matrices size : " << input_matrix_size_gb << " GB\n"
            << "Output matrix size  : " << output_matrix_size_gb << " GB\n"
            << "Required mem        : " << mem_gb << " GB"
            << std::endl;
        if (mem_gb > gpu_mem) {
            std::cerr << "[ERROR] GPU memory size is less than the matrix size" << std::endl;
            return 1;
        }


        std::cout << "SETUP:" << std::endl;
        const auto setup_tp0 = std::chrono::high_resolution_clock::now();

        std::cout << "  - Allocating memory: ";
        std::vector<float> h_A(input_size, 0.0f);
        std::vector<float> h_B(input_size, 0.0f);
        std::vector<float> h_C(output_size, 0.0f);
        const auto setup_tp1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> setup_dt1 = setup_tp1 - setup_tp0;
        std::cout << setup_dt1.count() << " ms (" << setup_dt1.count() << " ms total)" << std::endl;

        std::cout << "  - Initializing matrices: ";
        randomize_vector(h_A, seed);
        randomize_vector(h_B, seed+1);
        const auto setup_tp2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> setup_step_dt2 = setup_tp2 - setup_tp1;
        std::chrono::duration<double, std::milli> setup_total_dt2 = setup_tp2 - setup_tp0;
        std::cout << setup_step_dt2.count() << " ms (" << setup_total_dt2.count() << " ms total)" << std::endl;

        std::cout << "  - Creating GPU streams: ";
        cudaStream_t stream;
        cuda_check_error(cudaStreamCreate(&stream), "cudaStreamCreate");
        const auto setup_tp3 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> setup_step_dt3 = setup_tp3 - setup_tp2;
        std::chrono::duration<double, std::milli> setup_total_dt3 = setup_tp3 - setup_tp0;
        std::cout << setup_step_dt3.count() << " ms (" << setup_total_dt3.count() << " ms total)" << std::endl;

        std::cout << "  - Creating GPU events: ";
        cudaEvent_t e0, e1, e2, e3, e4, e5;
        cuda_check_error(cudaEventCreate(&e0), "cudaEventCreate");
        cuda_check_error(cudaEventCreate(&e1), "cudaEventCreate");
        cuda_check_error(cudaEventCreate(&e2), "cudaEventCreate");
        cuda_check_error(cudaEventCreate(&e3), "cudaEventCreate");
        cuda_check_error(cudaEventCreate(&e4), "cudaEventCreate");
        cuda_check_error(cudaEventCreate(&e5), "cudaEventCreate");
        const auto setup_tp4 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> setup_step_dt4 = setup_tp4 - setup_tp3;
        std::chrono::duration<double, std::milli> setup_total_dt4 = setup_tp4 - setup_tp0;
        std::cout << setup_step_dt4.count() << " ms (" << setup_total_dt4.count() << " ms total)" << std::endl;


        std::cout << "KERNEL:" << std::endl;
        const auto gpu_tp0 = std::chrono::high_resolution_clock::now();
        cuda_check_error(cudaEventRecord(e0, stream), "cudaEventRecord");

        const auto gpu_step_1 = "Allocate device memory";
        float *d_A, *d_B, *d_C;
        cuda_check_error(cudaMallocAsync(&d_A, input_size * sizeof(float), stream), "cudaMallocAsync");
        cuda_check_error(cudaMallocAsync(&d_B, input_size * sizeof(float), stream), "cudaMallocAsync");
        cuda_check_error(cudaMallocAsync(&d_C, output_size * sizeof(float), stream), "cudaMallocAsync");
        cuda_check_error(cudaEventRecord(e1, stream), "cudaEventRecord");
        std::chrono::high_resolution_clock::time_point gpu_tp1{};
        cudaStreamAddCallback(stream, report_completion_time_callback, &gpu_tp1, NULL_FLAGS);

        const auto gpu_step_2 = "Copy data to device";
        cuda_check_error(cudaMemcpyAsync(d_A, h_A.data(), input_size * sizeof(float), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync");
        cuda_check_error(cudaMemcpyAsync(d_B, h_B.data(), input_size * sizeof(float), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync");
        cuda_check_error(cudaEventRecord(e2, stream), "cudaEventRecord");
        std::chrono::high_resolution_clock::time_point gpu_tp2{};
        cudaStreamAddCallback(stream, report_completion_time_callback, &gpu_tp2, NULL_FLAGS);

        const auto gpu_step_3 = "Compute kernel";
        dim3 blockDim(16, 16);
        dim3 gridDim((M + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);
        matrixMultiply<<<gridDim, blockDim, 0, stream>>>(d_A, d_B, d_C, M, N, M);
        cuda_check_error(cudaEventRecord(e3, stream), "cudaEventRecord");
        std::chrono::high_resolution_clock::time_point gpu_tp3{};
        cuda_check_error(cudaStreamAddCallback(stream, report_completion_time_callback, &gpu_tp3, NULL_FLAGS), "cudaStreamAddCallback");

        const auto gpu_step_4 = "Copy result back to host";
        cuda_check_error(cudaMemcpyAsync(h_C.data(), d_C, output_size * sizeof(float), cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync");
        cuda_check_error(cudaEventRecord(e4, stream), "cudaEventRecord");
        std::chrono::high_resolution_clock::time_point gpu_tp4{};
        cuda_check_error(cudaStreamAddCallback(stream, report_completion_time_callback, &gpu_tp4, NULL_FLAGS), "cudaStreamAddCallback");

        const auto gpu_step_5 = "Free device memory";
        cuda_check_error(cudaFreeAsync(d_A, stream), "cudaFreeAsync");
        cuda_check_error(cudaFreeAsync(d_B, stream), "cudaFreeAsync");
        cuda_check_error(cudaFreeAsync(d_C, stream), "cudaFreeAsync");
        cuda_check_error(cudaEventRecord(e5, stream), "cudaEventRecord");
        std::chrono::high_resolution_clock::time_point gpu_tp5{};
        cuda_check_error(cudaStreamAddCallback(stream, report_completion_time_callback, &gpu_tp5, NULL_FLAGS), "cudaStreamAddCallback");

        // Wait for stream to finish
        cuda_check_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");


        // Print execution time
        constexpr int row_header_width = 22;
        constexpr int field_name_width = 25;
        float gpu_step_dt1 = 0.0f, gpu_step_dt2 = 0.0f, gpu_step_dt3 = 0.0f, gpu_step_dt4 = 0.0f, gpu_step_dt5 = 0.0f;
        float gpu_total_dt1 = 0.0f, gpu_total_dt2 = 0.0f, gpu_total_dt3 = 0.0f, gpu_total_dt4 = 0.0f, gpu_total_dt5 = 0.0f;

        std::chrono::duration<double, std::milli> chrono_step_dt1 = gpu_tp1 - gpu_tp0;
        std::chrono::duration<double, std::milli> chrono_total_dt1 = gpu_tp1 - gpu_tp0;
        cuda_check_error(cudaEventElapsedTime(&gpu_step_dt1, e0, e1), "cudaEventElapsedTime");
        cuda_check_error(cudaEventElapsedTime(&gpu_total_dt1, e0, e1), "cudaEventElapsedTime");
        std::cout << " - " << std::setw(row_header_width) << "cudaEventElapsedTime " << std::setw(field_name_width) << gpu_step_1 << ": " << chrono_step_dt1.count() << " ms (" << chrono_total_dt1.count() << " ms total)" << std::endl;
        std::cout << " - " << std::setw(row_header_width) << "std::chrono::duration " << std::setw(field_name_width) << gpu_step_1 << ": " << gpu_step_dt1 << " ms (" << gpu_total_dt1 << " ms total)" << std::endl;

        std::chrono::duration<double, std::milli> chrono_step_dt2 = gpu_tp2 - gpu_tp1;
        std::chrono::duration<double, std::milli> chrono_total_dt2 = gpu_tp2 - gpu_tp0;
        cuda_check_error(cudaEventElapsedTime(&gpu_step_dt2, e1, e2), "cudaEventElapsedTime");
        cuda_check_error(cudaEventElapsedTime(&gpu_total_dt2, e0, e2), "cudaEventElapsedTime");
        std::cout << " - " << std::setw(row_header_width) << "cudaEventElapsedTime " << std::setw(field_name_width) << gpu_step_2 << ": " << chrono_step_dt2.count() << " ms (" << chrono_total_dt2.count() << " ms total)" << std::endl;
        std::cout << " - " << std::setw(row_header_width) << "std::chrono::duration " << std::setw(field_name_width) << gpu_step_2 << ": " << gpu_step_dt2 << " ms (" << gpu_total_dt2 << " ms total)" << std::endl;

        std::chrono::duration<double, std::milli> chrono_step_dt3 = gpu_tp3 - gpu_tp2;
        std::chrono::duration<double, std::milli> chrono_total_dt3 = gpu_tp3 - gpu_tp0;
        cuda_check_error(cudaEventElapsedTime(&gpu_step_dt3, e2, e3), "cudaEventElapsedTime");
        cuda_check_error(cudaEventElapsedTime(&gpu_total_dt3, e0, e3), "cudaEventElapsedTime");
        std::cout << " - " << std::setw(row_header_width) << "cudaEventElapsedTime " << std::setw(field_name_width) << gpu_step_3 << ": " << chrono_step_dt3.count() << " ms (" << chrono_total_dt3.count() << " ms total)" << std::endl;
        std::cout << " - " << std::setw(row_header_width) << "std::chrono::duration " << std::setw(field_name_width) << gpu_step_3 << ": " << gpu_step_dt3 << " ms (" << gpu_total_dt3 << " ms total)" << std::endl;

        std::chrono::duration<double, std::milli> chrono_step_dt4 = gpu_tp4 - gpu_tp3;
        std::chrono::duration<double, std::milli> chrono_total_dt4 = gpu_tp4 - gpu_tp0;
        cuda_check_error(cudaEventElapsedTime(&gpu_step_dt4, e3, e4), "cudaEventElapsedTime");
        cuda_check_error(cudaEventElapsedTime(&gpu_total_dt4, e0, e4), "cudaEventElapsedTime");
        std::cout << " - " << std::setw(row_header_width) << "cudaEventElapsedTime " << std::setw(field_name_width) << gpu_step_4 << ": " << chrono_step_dt4.count() << " ms (" << chrono_total_dt4.count() << " ms total)" << std::endl;
        std::cout << " - " << std::setw(row_header_width) << "std::chrono::duration " << std::setw(field_name_width) << gpu_step_4 << ": " << gpu_step_dt4 << " ms (" << gpu_total_dt4 << " ms total)" << std::endl;

        std::chrono::duration<double, std::milli> chrono_step_dt5 = gpu_tp5 - gpu_tp4;
        std::chrono::duration<double, std::milli> chrono_total_dt5 = gpu_tp5 - gpu_tp0;
        cuda_check_error(cudaEventElapsedTime(&gpu_step_dt5, e4, e5), "cudaEventElapsedTime");
        cuda_check_error(cudaEventElapsedTime(&gpu_total_dt5, e0, e5), "cudaEventElapsedTime");
        std::cout << " - " << std::setw(row_header_width) << "cudaEventElapsedTime " << std::setw(field_name_width) << gpu_step_5 << ": " << chrono_step_dt5.count() << " ms (" << chrono_total_dt5.count() << " ms total)" << std::endl;
        std::cout << " - " << std::setw(row_header_width) << "std::chrono::duration " << std::setw(field_name_width) << gpu_step_5 << ": " << gpu_step_dt5 << " ms (" << gpu_total_dt5 << " ms total)" << std::endl;

        const auto cpu_tp0 = std::chrono::high_resolution_clock::now();

        constexpr int check_field_width = 26;
        std::cout << "CHECK WITH CPU:" << std::endl;
        const auto cpu_step_1 = "Convert data to Eigen";
        const Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> A{h_A.data(), M, N};
        const Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> B{h_B.data(), N, M};
        const Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> C_gpu{h_C.data(), M, M};
        const auto cpu_tp1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> cpu_step_dt1 = cpu_tp1 - cpu_tp0;
        std::chrono::duration<double, std::milli> cpu_total_dt1 = cpu_tp1 - cpu_tp0;
        std::cout << " - " << std::setw(check_field_width) << cpu_step_1 << ": " << cpu_step_dt1.count() << " ms (" << cpu_total_dt1.count() << " ms total)" << std::endl;

        const auto cpu_step_2 = "Compute result with Eigen";
        const auto C_cpu = (A * B).eval();
        const auto cpu_tp2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> cpu_step_dt2 = cpu_tp2 - cpu_tp1;
        std::chrono::duration<double, std::milli> cpu_total_dt2 = cpu_tp2 - cpu_tp0;
        std::cout << " - " << std::setw(check_field_width) << cpu_step_2 << ": " << cpu_step_dt2.count() << " ms (" << cpu_total_dt2.count() << " ms total)" << std::endl;

        const auto cpu_step_3 = "Compute error matrix";
        const auto E = (C_gpu - C_cpu).eval();
        const auto cpu_tp3 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> cpu_step_dt3 = cpu_tp3 - cpu_tp2;
        std::chrono::duration<double, std::milli> cpu_total_dt3 = cpu_tp3 - cpu_tp0;
        std::cout << " - " << std::setw(check_field_width) << cpu_step_3 << ": " << cpu_step_dt3.count() << " ms (" << cpu_total_dt3.count() << " ms total)" << std::endl;

        const auto cpu_step_4 = "Compute max error";
        const auto E_max = E.cwiseAbs().maxCoeff();
        const auto cpu_tp4 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> cpu_step_dt4 = cpu_tp4 - cpu_tp3;
        std::chrono::duration<double, std::milli> cpu_total_dt4 = cpu_tp4 - cpu_tp0;
        std::cout << " - " << std::setw(check_field_width) << cpu_step_4 << ": " << cpu_step_dt4.count() << " ms (" << cpu_total_dt4.count() << " ms total)" << std::endl;

        const auto cpu_step_5 = "Compute max relative error";
        const auto E_max_pct = E_max / C_cpu.cwiseAbs().maxCoeff() * 100.0f;
        const auto cpu_tp5 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> cpu_step_dt5 = cpu_tp5 - cpu_tp4;
        std::chrono::duration<double, std::milli> cpu_total_dt5 = cpu_tp5 - cpu_tp0;
        std::cout << " - " << std::setw(check_field_width) << cpu_step_5 << ": " << cpu_step_dt5.count() << " ms (" << cpu_total_dt5.count() << " ms total)" << std::endl;

        std::cout << "Max error: " << E_max << " (" << E_max_pct << " % )" << std::endl;

        if (verbose) {
            const Eigen::IOFormat clean_matrix_format(4, 0, ", ", "\n", "  [", "]");
            std::cout << "A :\n" << A.format(clean_matrix_format) << std::endl;
            std::cout << "B :\n" << B.format(clean_matrix_format) << std::endl;
            std::cout << "C_gpu :\n" << C_gpu.format(clean_matrix_format) << std::endl;
            std::cout << "C_cpu :\n" << C_cpu.format(clean_matrix_format) << std::endl;
            std::cout << "E :\n" << E.format(clean_matrix_format) << std::endl;
        }

        const auto tp_done = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> total_dt = tp_done - setup_tp0;
        std::cout << "DONE: " << total_dt.count() << " ms total" << std::endl;


    } catch (const cxxopts::exceptions::exception& e) {
        std::cerr << "Error parsing options: " << e.what() << std::endl;
        std::cout << options.help() << std::endl;
        return 1;
    }

    return 0;
}
