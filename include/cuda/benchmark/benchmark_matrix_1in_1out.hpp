// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: include/cuda/benchmark/benchmark_matrix_1in_1out.hpp

#pragma once


#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <Eigen/Dense>

#include <cuda_runtime.h>
#include <cxxopts.hpp>

#include "cuda/random.hpp"
#include "cuda/check_errors.hpp"
#include "cuda/cuda_utils.hpp"
#include "cuda/kernel_api/matrix_1in_1out.hpp"

template <MATRIX_KERNEL_1IN_1OUT Matrix_Kernel_1In_1Out>
class Benchmark_Matrix_1In_1Out {
    public:
    using Kernel_spec = typename Matrix_Kernel_1In_1Out::Kernel_spec;
    using Number = typename Matrix_Kernel_1In_1Out::Number;
    using Printable_Number = std::conditional_t<std::is_same_v<Number, __half>, float, Number>;

    const Kernel_spec spec;
    const int seed;
    const int gpu_mem;
    const bool verbose;
    const bool errors;
    const bool force;
    const std::string init_method;

    Matrix_Kernel_1In_1Out kernel;

    template <typename... Args>
    Benchmark_Matrix_1In_1Out(
        const Kernel_spec spec,
        const cxxopts::Options& options,
        const cxxopts::ParseResult& options_parsed,
        Args&... args
    ) : spec(spec),
        seed(options_parsed["seed"].as<long>()),
        gpu_mem(options_parsed["gpumem"].as<long>()),
        verbose(options_parsed["verbose"].as<bool>()),
        errors(options_parsed["errors"].as<bool>()),
        force(options_parsed["force"].as<bool>()),
        init_method(options_parsed["init-method"].as<std::string>()),
        kernel(spec, args...)
    {
        // Handle help option first
        if (options_parsed.count("help")) {
            std::cout << options.help() << std::endl;
            exit(0);
        }
        if (verbose && (
            (spec.n_rows_A_ > 10000 || spec.n_cols_A_ > 1000)
            || (spec.n_rows_C_ > 10000 || spec.n_cols_C_ > 1000)
        )) {
            std::cerr << "WARNING: verbose mode is enabled and the input matrices are large."
            << "This will print the entire matrices to the console." << std::endl;
            if (!force) {
                std::cerr << "Use --force to override." << std::endl
                          << "[ERROR] matrices too big for verbose mode" << std::endl;
                exit(1);
            }
        }
    }

    int run() {
        const size_t size_A = size_t(spec.n_rows_A_) * size_t(spec.n_cols_A_);
        const size_t size_C = size_t(spec.n_rows_C_) * size_t(spec.n_cols_C_);
        const size_t size_temp = size_t(spec.n_rows_temp_) * size_t(spec.n_cols_temp_);
        const size_t size_A_bytes = size_A * sizeof(Number);
        const size_t size_C_bytes = size_C * sizeof(Number);
        const size_t size_temp_bytes = size_temp * sizeof(Number);
        const size_t input_size_bytes = size_A_bytes;
        const size_t output_size_bytes = size_C_bytes;
        const size_t temp_size_bytes = size_temp_bytes;
        const size_t mem_size_bytes = input_size_bytes + output_size_bytes + temp_size_bytes;
        constexpr float GB = 1024.0f * 1024.0f * 1024.0f;
        const float input_size_gb = input_size_bytes / GB;
        const float output_size_gb = output_size_bytes / GB;
        const float temp_size_gb = temp_size_bytes / GB;
        const float mem_gb = mem_size_bytes / GB;

        const auto [is_random, is_increasing, is_decreasing] = [&](){
            if (init_method == "random") {
                return std::tuple{true, false, false};
            } else if (init_method == "increasing") {
                return std::tuple{false, true, false};
            } else if (init_method == "decreasing") {
                return std::tuple{false, false, true};
            } else {
                std::cerr << "[ERROR] Invalid initialization method" << std::endl;
                exit(1);
            }
        }();

        std::cout
            << "Input matrix A dimensions   : " << spec.n_rows_A_ << "x" << spec.n_cols_A_ << "\n"
            << "Output matrix dimensions    : " << spec.n_rows_C_ << "x" << spec.n_cols_C_ << "\n"
            << "Temp matrix dimensions      : " << spec.n_rows_temp_ << "x" << spec.n_cols_temp_ << "\n"
            << "Input size                  : " << input_size_gb << " GB (" << input_size_bytes << " bytes)\n"
            << "Output size                 : " << output_size_gb << " GB (" << output_size_bytes << " bytes)\n"
            << "Temp size                   : " << temp_size_gb << " GB (" << temp_size_bytes << " bytes)\n"
            << "Required memory             : " << mem_gb << " GB (" << mem_size_bytes << " bytes)\n"
            << std::endl;
        if (mem_gb > gpu_mem) {
            std::cerr << "[ERROR] GPU memory size is less than the required size" << std::endl;
            return 1;
        }

        std::cout << "SETUP:" << std::endl;
        const auto setup_tp0 = std::chrono::high_resolution_clock::now();

        std::cout << "  - Allocating memory: ";
        std::vector<Number> vec_A(size_A, 0);
        std::vector<Number> vec_C(size_C, 0);
        std::vector<Number> vec_temp(size_temp, 0);
        const auto setup_tp1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> setup_dt1 = setup_tp1 - setup_tp0;
        std::cout << setup_dt1.count() << " ms (" << setup_dt1.count() << " ms total)" << std::endl;

        std::cout << "  - Initializing matrices: ";
        if (is_random) {
            std::cout << "  - Randomizing matrices: ";
            randomize_vector(vec_A, seed);
        } else if (is_increasing) {
            for (size_t i = 0; i < size_A; ++i) vec_A[i] = Number(i);
        } else if (is_decreasing) {
            for (size_t i = 0; i < size_A; ++i) vec_A[i] = Number(size_A - i);
        } else {
            std::cerr << "[ERROR] Invalid initialization method" << std::endl;
            exit(1);
        }
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

        std::cout << "Matrix_Kernel_1In_1Out:" << std::endl;
        const auto gpu_tp0 = std::chrono::high_resolution_clock::now();
        cuda_check_error(cudaEventRecord(e0, stream), "cudaEventRecord");

        const auto gpu_step_1 = "Allocate device memory";
        Number* gpu_data_A = nullptr;
        Number* gpu_data_C = nullptr;
        Number* gpu_data_temp = nullptr;

        cuda_check_error(cudaMallocAsync(&gpu_data_A, size_A_bytes, stream), "cudaMallocAsync");
        cuda_check_error(cudaMallocAsync(&gpu_data_C, size_C_bytes, stream), "cudaMallocAsync");
        if (size_temp_bytes > 0) {
            cuda_check_error(cudaMallocAsync(&gpu_data_temp, size_temp_bytes, stream), "cudaMallocAsync");
        }
        cuda_check_error(cudaEventRecord(e1, stream), "cudaEventRecord");
        std::chrono::high_resolution_clock::time_point gpu_tp1{};
        cuda_check_error(cudaStreamAddCallback(stream, report_completion_time_callback, &gpu_tp1, NULL_FLAGS), "cudaStreamAddCallback");

        const auto gpu_step_2 = "Copy data to device";
        cuda_check_error(cudaMemcpyAsync(gpu_data_A, vec_A.data(), size_A_bytes, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync");
        cuda_check_error(cudaEventRecord(e2, stream), "cudaEventRecord");
        std::chrono::high_resolution_clock::time_point gpu_tp2{};
        cuda_check_error(cudaStreamAddCallback(stream, report_completion_time_callback, &gpu_tp2, NULL_FLAGS), "cudaStreamAddCallback");

        const auto gpu_step_3 = "Compute kernel";
        kernel.run_device_kernel(gpu_data_A, gpu_data_C, gpu_data_temp, stream);
        cuda_check_error(cudaEventRecord(e3, stream), "cudaEventRecord");
        std::chrono::high_resolution_clock::time_point gpu_tp3{};
        cuda_check_error(cudaStreamAddCallback(stream, report_completion_time_callback, &gpu_tp3, NULL_FLAGS), "cudaStreamAddCallback");

        const auto gpu_step_4 = "Copy result back to host";
        cuda_check_error(cudaMemcpyAsync(vec_C.data(), gpu_data_C, size_C_bytes, cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync");
        if (size_temp_bytes > 0) {
            cuda_check_error(cudaMemcpyAsync(vec_temp.data(), gpu_data_temp, size_temp_bytes, cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync");
        }
        cuda_check_error(cudaEventRecord(e4, stream), "cudaEventRecord");
        std::chrono::high_resolution_clock::time_point gpu_tp4{};
        cuda_check_error(cudaStreamAddCallback(stream, report_completion_time_callback, &gpu_tp4, NULL_FLAGS), "cudaStreamAddCallback");

        const auto gpu_step_5 = "Free device memory";
        cuda_check_error(cudaFreeAsync(gpu_data_A, stream), "cudaFreeAsync");
        cuda_check_error(cudaFreeAsync(gpu_data_C, stream), "cudaFreeAsync");
        if (size_temp_bytes > 0) {
            cuda_check_error(cudaFreeAsync(gpu_data_temp, stream), "cudaFreeAsync");
        }
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

        // Clean up
        cuda_check_error(cudaEventDestroy(e0), "cudaEventDestroy");
        cuda_check_error(cudaEventDestroy(e1), "cudaEventDestroy");
        cuda_check_error(cudaEventDestroy(e2), "cudaEventDestroy");
        cuda_check_error(cudaEventDestroy(e3), "cudaEventDestroy");
        cuda_check_error(cudaEventDestroy(e4), "cudaEventDestroy");
        cuda_check_error(cudaEventDestroy(e5), "cudaEventDestroy");
        cuda_check_error(cudaStreamDestroy(stream), "cudaStreamDestroy");

        const auto cpu_tp0 = std::chrono::high_resolution_clock::now();

        constexpr int check_field_width = 26;
        std::cout << "CHECK WITH CPU:" << std::endl;
        const auto cpu_step_1 = "Convert data to Eigen";
        const Eigen::Map<Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> A{vec_A.data(), spec.n_rows_A_, spec.n_cols_A_};
        const Eigen::Map<Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> C_gpu{vec_C.data(), spec.n_rows_C_, spec.n_cols_C_};
        const auto cpu_tp1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> cpu_step_dt1 = cpu_tp1 - cpu_tp0;
        std::chrono::duration<double, std::milli> cpu_total_dt1 = cpu_tp1 - cpu_tp0;
        std::cout << " - " << std::setw(check_field_width) << cpu_step_1 << ": " << cpu_step_dt1.count() << " ms (" << cpu_total_dt1.count() << " ms total)" << std::endl;

        const auto cpu_step_2 = "Compute result with Eigen";
        const auto C_cpu = kernel.run_host_kernel(A);
        const auto cpu_tp2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> cpu_step_dt2 = cpu_tp2 - cpu_tp1;
        std::chrono::duration<double, std::milli> cpu_total_dt2 = cpu_tp2 - cpu_tp0;
        std::cout << " - " << std::setw(check_field_width) << cpu_step_2 << ": " << cpu_step_dt2.count() << " ms (" << cpu_total_dt2.count() << " ms total)" << std::endl;

        const auto cpu_step_3 = "Compute error matrix";
        const auto E = (C_gpu - C_cpu).eval();
        const auto E_pct = E.cwiseAbs().template cast<double>().array() / C_cpu.cwiseAbs().template cast<double>().array();
        const auto cpu_tp3 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> cpu_step_dt3 = cpu_tp3 - cpu_tp2;
        std::chrono::duration<double, std::milli> cpu_total_dt3 = cpu_tp3 - cpu_tp0;
        std::cout << " - " << std::setw(check_field_width) << cpu_step_3 << ": " << cpu_step_dt3.count() << " ms (" << cpu_total_dt3.count() << " ms total)" << std::endl;

        const auto cpu_step_4 = "Compute max error";
        size_t E_max_row, E_max_col, E_pct_max_row, E_pct_max_col;
        const double E_max = E.cwiseAbs().maxCoeff(&E_max_row, &E_max_col);
        const auto E_max_pct = E_pct.maxCoeff(&E_pct_max_row, &E_pct_max_col);
        const auto cpu_tp4 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> cpu_step_dt4 = cpu_tp4 - cpu_tp3;
        std::chrono::duration<double, std::milli> cpu_total_dt4 = cpu_tp4 - cpu_tp0;
        std::cout << " - " << std::setw(check_field_width) << cpu_step_4 << ": " << cpu_step_dt4.count() << " ms (" << cpu_total_dt4.count() << " ms total)" << std::endl;

        if (errors) {
            std::cout << "Non-zero error elements:\n";
            bool found_errors = false;
            for (int i = 0; i < E.rows(); ++i) {
                for (int j = 0; j < E.cols(); ++j) {
                    if (E(i, j) != Number(0)) {
                        found_errors = true;
                        std::cout << "(" << i << ", " << j << "): "
                                  << "C_gpu=" << static_cast<Printable_Number>(C_gpu(i, j)) << ", "
                                  << "C_cpu=" << static_cast<Printable_Number>(C_cpu(i, j)) << ", "
                                  << "E=" << static_cast<Printable_Number>(E(i, j)) << "\n";
                    }
                }
            }
            if (!found_errors) {
                std::cout << "No non-zero error elements found\n";
            }
        }

        if (verbose) {
            const Eigen::IOFormat eigen_format(4, 0, ", ", "\n", "  [", "]");
            std::cout << "A    :\n";
            std::cout << A.template cast<Printable_Number>().format(eigen_format) << std::endl;
            std::cout << "C_gpu:\n";
            std::cout << C_gpu.template cast<Printable_Number>().format(eigen_format) << std::endl;
            std::cout << "C_cpu:\n";
            std::cout << C_cpu.template cast<Printable_Number>().format(eigen_format) << std::endl;
            if ((spec.n_rows_temp_ > 0) && (spec.n_cols_temp_ > 0)) {
                const Eigen::Map<Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> tmp_gpu{
                    vec_temp.data(), spec.n_rows_temp_, spec.n_cols_temp_
                };
                std::cout << "tmp  :\n";
                std::cout << tmp_gpu.template cast<Printable_Number>().format(eigen_format) << std::endl;
            }
            std::cout << "E    :\n";
            std::cout << E.template cast<Printable_Number>().format(eigen_format) << std::endl;
            std::cout << "E_pct:\n";
            std::cout << E_pct.template cast<Printable_Number>().format(eigen_format) << std::endl;
        }

        const auto tp_done = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> total_dt = tp_done - setup_tp0;
        std::cout << "DONE: " << total_dt.count() << " ms total" << std::endl;
        std::cout << "Max error     : " << E_max << " at (" << E_max_row << ", " << E_max_col << ")" << std::endl;
        std::cout << "Max error pct : " << E_max_pct << " at (" << E_pct_max_row << ", " << E_pct_max_col << ")" << std::endl;
        std::cout << "Gross speedup : " << (cpu_step_dt2.count()/gpu_step_dt3) << std::endl;
        std::cout << "Net speedup   : " << (cpu_total_dt2.count()/gpu_total_dt5) << std::endl;

        return 0;
    }
};
