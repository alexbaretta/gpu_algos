// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: include/common/benchmark.h

#pragma once

#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <Eigen/Dense>

#include <cuda_runtime.h>
#include <cxxopts.hpp>

#include "common/random.h"
#include "cuda/check_errors.h"
#include "cuda/cuda_utils.h"
#include "cuda/kernel_api.h"

#ifndef _OPENMP
static_assert(false, "OpenMP is not supported");
#endif

constexpr int DEFAULT_GPU_MEM = 16; // GPU memory size in GB
constexpr int DEFAULT_SEED = 42;

void add_benchmark_options(cxxopts::Options& options);

template <Kernel KERNEL>
class Benchmark {
    public:
    using KERNEL_SPEC = typename KERNEL::KERNEL_SPEC;
    using NUMBER = typename KERNEL::NUMBER;
    using PRINTABLE_NUMBER = std::conditional_t<std::is_same_v<NUMBER, __half>, float, NUMBER>;

    const KERNEL_SPEC spec;
    const int seed;
    const int gpu_mem;
    const bool verbose;

    // Parse CLI options
    Benchmark(
        const KERNEL_SPEC spec,
        const cxxopts::ParseResult& options_parsed
    ) : spec(spec),
        seed(options_parsed["seed"].as<int>()),
        gpu_mem(options_parsed["gpumem"].as<int>()),
        verbose(options_parsed["verbose"].as<bool>())
    {}

    int run() {
        const int size_A = spec.n_rows_A_ * spec.n_cols_A_;
        const int size_B = spec.n_rows_B_ * spec.n_cols_B_;
        const int size_C = spec.n_rows_C_ * spec.n_cols_C_;
        const int size_A_bytes = size_A * sizeof(NUMBER);
        const int size_B_bytes = size_B * sizeof(NUMBER);
        const int size_C_bytes = size_C * sizeof(NUMBER);
        const int input_size_bytes = size_A_bytes + size_B_bytes;
        const int output_size_bytes = size_C_bytes;
        constexpr float GB = 1024.0f * 1024.0f * 1024.0f;
        const float input_size_gb = input_size_bytes / GB;
        const float output_size_gb = output_size_bytes / GB;
        const float mem_gb = 2*input_size_gb + output_size_gb;
        std::cout
            << "Input matrices dimensions   : " << spec.n_rows_A_ << "x" << spec.n_cols_A_ << " * " << spec.n_cols_B_ << "x" << spec.n_rows_B_ << "\n"
            << "Input size                  : " << input_size_gb << " GB\n"
            << "Output size                 : " << output_size_gb << " GB\n"
            << "Required memory             : " << mem_gb << " GB"
            << std::endl;
        if (mem_gb > gpu_mem) {
            std::cerr << "[ERROR] GPU memory size is less than the matrix size" << std::endl;
            return 1;
        }


        std::cout << "SETUP:" << std::endl;
        const auto setup_tp0 = std::chrono::high_resolution_clock::now();

        std::cout << "  - Allocating memory: ";
        std::vector<NUMBER> vec_A(size_A, 0.0f);
        std::vector<NUMBER> vec_B(size_B, 0.0f);
        std::vector<NUMBER> vec_C(size_C, 0.0f);
        const auto setup_tp1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> setup_dt1 = setup_tp1 - setup_tp0;
        std::cout << setup_dt1.count() << " ms (" << setup_dt1.count() << " ms total)" << std::endl;

        std::cout << "  - Initializing matrices: ";
        randomize_vector(vec_A, seed);
        randomize_vector(vec_B, seed+1);
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
        NUMBER *gpu_data_A = nullptr, *gpu_data_B = nullptr, *gpu_data_C = nullptr;
        cuda_check_error(cudaMallocAsync(&gpu_data_A, size_A_bytes, stream), "cudaMallocAsync");
        cuda_check_error(cudaMallocAsync(&gpu_data_B, size_B_bytes, stream), "cudaMallocAsync");
        cuda_check_error(cudaMallocAsync(&gpu_data_C, size_C_bytes, stream), "cudaMallocAsync");
        cuda_check_error(cudaEventRecord(e1, stream), "cudaEventRecord");
        std::chrono::high_resolution_clock::time_point gpu_tp1{};
        cudaStreamAddCallback(stream, report_completion_time_callback, &gpu_tp1, NULL_FLAGS);

        const auto gpu_step_2 = "Copy data to device";
        cuda_check_error(cudaMemcpyAsync(gpu_data_A, vec_A.data(), size_A_bytes, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync");
        cuda_check_error(cudaMemcpyAsync(gpu_data_B, vec_B.data(), size_B_bytes, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync");
        cuda_check_error(cudaEventRecord(e2, stream), "cudaEventRecord");
        std::chrono::high_resolution_clock::time_point gpu_tp2{};
        cudaStreamAddCallback(stream, report_completion_time_callback, &gpu_tp2, NULL_FLAGS);

        const auto gpu_step_3 = "Compute kernel";
        KERNEL kernel(spec, gpu_data_A, gpu_data_B, gpu_data_C, stream);
        kernel.run_kernel();
        cuda_check_error(cudaEventRecord(e3, stream), "cudaEventRecord");
        std::chrono::high_resolution_clock::time_point gpu_tp3{};
        cuda_check_error(cudaStreamAddCallback(stream, report_completion_time_callback, &gpu_tp3, NULL_FLAGS), "cudaStreamAddCallback");

        const auto gpu_step_4 = "Copy result back to host";
        cuda_check_error(cudaMemcpyAsync(vec_C.data(), gpu_data_C, size_C_bytes, cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync");
        cuda_check_error(cudaEventRecord(e4, stream), "cudaEventRecord");
        std::chrono::high_resolution_clock::time_point gpu_tp4{};
        cuda_check_error(cudaStreamAddCallback(stream, report_completion_time_callback, &gpu_tp4, NULL_FLAGS), "cudaStreamAddCallback");

        const auto gpu_step_5 = "Free device memory";
        cuda_check_error(cudaFreeAsync(gpu_data_A, stream), "cudaFreeAsync");
        cuda_check_error(cudaFreeAsync(gpu_data_B, stream), "cudaFreeAsync");
        cuda_check_error(cudaFreeAsync(gpu_data_C, stream), "cudaFreeAsync");
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
        std::cout << "1 - " << std::setw(row_header_width) << "cudaEventElapsedTime " << std::setw(field_name_width) << gpu_step_1 << ": " << chrono_step_dt1.count() << " ms (" << chrono_total_dt1.count() << " ms total)" << std::endl;
        std::cout << "1 - " << std::setw(row_header_width) << "std::chrono::duration " << std::setw(field_name_width) << gpu_step_1 << ": " << gpu_step_dt1 << " ms (" << gpu_total_dt1 << " ms total)" << std::endl;

        std::chrono::duration<double, std::milli> chrono_step_dt2 = gpu_tp2 - gpu_tp1;
        std::chrono::duration<double, std::milli> chrono_total_dt2 = gpu_tp2 - gpu_tp0;
        cuda_check_error(cudaEventElapsedTime(&gpu_step_dt2, e1, e2), "cudaEventElapsedTime");
        cuda_check_error(cudaEventElapsedTime(&gpu_total_dt2, e0, e2), "cudaEventElapsedTime");
        std::cout << "2 - " << std::setw(row_header_width) << "cudaEventElapsedTime " << std::setw(field_name_width) << gpu_step_2 << ": " << chrono_step_dt2.count() << " ms (" << chrono_total_dt2.count() << " ms total)" << std::endl;
        std::cout << "2 - " << std::setw(row_header_width) << "std::chrono::duration " << std::setw(field_name_width) << gpu_step_2 << ": " << gpu_step_dt2 << " ms (" << gpu_total_dt2 << " ms total)" << std::endl;

        std::chrono::duration<double, std::milli> chrono_step_dt3 = gpu_tp3 - gpu_tp2;
        std::chrono::duration<double, std::milli> chrono_total_dt3 = gpu_tp3 - gpu_tp0;
        cuda_check_error(cudaEventElapsedTime(&gpu_step_dt3, e2, e3), "cudaEventElapsedTime");
        cuda_check_error(cudaEventElapsedTime(&gpu_total_dt3, e0, e3), "cudaEventElapsedTime");
        std::cout << "3 - " << std::setw(row_header_width) << "cudaEventElapsedTime " << std::setw(field_name_width) << gpu_step_3 << ": " << chrono_step_dt3.count() << " ms (" << chrono_total_dt3.count() << " ms total)" << std::endl;
        std::cout << "3 - " << std::setw(row_header_width) << "std::chrono::duration " << std::setw(field_name_width) << gpu_step_3 << ": " << gpu_step_dt3 << " ms (" << gpu_total_dt3 << " ms total)" << std::endl;

        std::chrono::duration<double, std::milli> chrono_step_dt4 = gpu_tp4 - gpu_tp3;
        std::chrono::duration<double, std::milli> chrono_total_dt4 = gpu_tp4 - gpu_tp0;
        cuda_check_error(cudaEventElapsedTime(&gpu_step_dt4, e3, e4), "cudaEventElapsedTime");
        cuda_check_error(cudaEventElapsedTime(&gpu_total_dt4, e0, e4), "cudaEventElapsedTime");
        std::cout << "4 - " << std::setw(row_header_width) << "cudaEventElapsedTime " << std::setw(field_name_width) << gpu_step_4 << ": " << chrono_step_dt4.count() << " ms (" << chrono_total_dt4.count() << " ms total)" << std::endl;
        std::cout << "4 - " << std::setw(row_header_width) << "std::chrono::duration " << std::setw(field_name_width) << gpu_step_4 << ": " << gpu_step_dt4 << " ms (" << gpu_total_dt4 << " ms total)" << std::endl;

        std::chrono::duration<double, std::milli> chrono_step_dt5 = gpu_tp5 - gpu_tp4;
        std::chrono::duration<double, std::milli> chrono_total_dt5 = gpu_tp5 - gpu_tp0;
        cuda_check_error(cudaEventElapsedTime(&gpu_step_dt5, e4, e5), "cudaEventElapsedTime");
        cuda_check_error(cudaEventElapsedTime(&gpu_total_dt5, e0, e5), "cudaEventElapsedTime");
        std::cout << "5 - " << std::setw(row_header_width) << "cudaEventElapsedTime " << std::setw(field_name_width) << gpu_step_5 << ": " << chrono_step_dt5.count() << " ms (" << chrono_total_dt5.count() << " ms total)" << std::endl;
        std::cout << "5 - " << std::setw(row_header_width) << "std::chrono::duration " << std::setw(field_name_width) << gpu_step_5 << ": " << gpu_step_dt5 << " ms (" << gpu_total_dt5 << " ms total)" << std::endl;

        const auto cpu_tp0 = std::chrono::high_resolution_clock::now();

        constexpr int check_field_width = 26;
        std::cout << "CHECK WITH CPU:" << std::endl;
        const auto cpu_step_1 = "Convert data to Eigen";
        const Eigen::Map<Eigen::Matrix<NUMBER, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> A{vec_A.data(), spec.n_rows_A_, spec.n_cols_A_};
        const Eigen::Map<Eigen::Matrix<NUMBER, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> B{vec_B.data(), spec.n_rows_B_, spec.n_cols_B_};
        const Eigen::Map<Eigen::Matrix<NUMBER, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> C_gpu{vec_C.data(), spec.n_rows_C_, spec.n_cols_C_};
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
        const double E_max = E.cwiseAbs().maxCoeff();
        const double C_cpu_max = C_cpu.cwiseAbs().maxCoeff();
        const auto E_max_pct = (E_max / C_cpu_max) * 100.0f;
        const auto cpu_tp4 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> cpu_step_dt4 = cpu_tp4 - cpu_tp3;
        std::chrono::duration<double, std::milli> cpu_total_dt4 = cpu_tp4 - cpu_tp0;
        std::cout << " - " << std::setw(check_field_width) << cpu_step_4 << ": " << cpu_step_dt4.count() << " ms (" << cpu_total_dt4.count() << " ms total)" << std::endl;

        std::cout << "Max error: " << E_max << " (" << E_max_pct << " % )" << std::endl;

        if (verbose) {
            const Eigen::IOFormat clean_matrix_format(4, 0, ", ", "\n", "  [", "]");
            std::cout << "A :\n" << A.template cast<PRINTABLE_NUMBER>().format(clean_matrix_format) << std::endl;
            std::cout << "B :\n" << B.template cast<PRINTABLE_NUMBER>().format(clean_matrix_format) << std::endl;
            std::cout << "C_gpu :\n" << C_gpu.template cast<PRINTABLE_NUMBER>().format(clean_matrix_format) << std::endl;
            std::cout << "C_cpu :\n" << C_cpu.template cast<PRINTABLE_NUMBER>().format(clean_matrix_format) << std::endl;
            std::cout << "E :\n" << E.template cast<PRINTABLE_NUMBER>().format(clean_matrix_format) << std::endl;
        }

        const auto tp_done = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> total_dt = tp_done - setup_tp0;
        std::cout << "DONE: " << total_dt.count() << " ms total" << std::endl;
        std::cout << "Gross speedup : " << (cpu_step_dt2.count()/gpu_step_dt3) << std::endl;
        std::cout << "Net speedup   : " << (cpu_total_dt2.count()/gpu_total_dt5) << std::endl;
        return 0;
    }
};
