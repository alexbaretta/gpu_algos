/*
    Copyright (c) 2025 Alessandro Baretta <alex@baretta.com>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/


// source path: include/cuda/benchmark/benchmark_matrix_1inout.cuh

#pragma once

#include <concepts>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <Eigen/Dense>

#include <cuda_runtime.h>
#include <cxxopts.hpp>

#include "common/random.hpp"
#include "cuda/eigen_utils.cuh"
#include "cuda/check_errors.cuh"
#include "cuda/cuda_utils.cuh"
#include "cuda/kernel_api.cuh"

template <MATRIX_KERNEL_1INOUT Matrix_Kernel_1Inout>
class Benchmark_Matrix_1Inout {
    public:
    using Kernel_spec = typename Matrix_Kernel_1Inout::Kernel_spec;
    using Kernel = Matrix_Kernel_1Inout;
    using Numbers = detect::Numbers<Kernel>;
    using NumberA = typename Numbers::A;
    using NumberB = typename Numbers::B;
    using NumberC = typename Numbers::C;
    using NumberD = typename Numbers::D;
    using NumberE = typename Numbers::E;
    using NumberTemp = typename Numbers::Temp;
    using Tolerance = typename Numbers::Tolerance;

    const Kernel_spec spec;
    const int seed;
    const int gpu_mem;
    const bool verbose;
    const bool errors;
    const bool force;
    const std::string init_method;
    const int tol_bits;

    Matrix_Kernel_1Inout kernel;

    template <typename... Args>
    Benchmark_Matrix_1Inout(
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
        tol_bits(std::integral<typename Tolerance::type> ? 0 : options_parsed["tol-bits"].as<int>()),
        kernel(spec, args...)
    {
        // Handle help option first
        if (options_parsed.count("help")) {
            std::cout << options.help() << std::endl;
            exit(0);
        }
        if (std::integral<typename Tolerance::type> && options_parsed.count("tol-bits")) {
            std::cout << "[ERROR] You may not use --tol-bits with integral types, which have no tolerance" << std::endl;
            std::cout << options.help() << std::endl;
            exit(1);
        }
        if (verbose && (
            (spec.n_rows_A_ > 10000 || spec.n_cols_A_ > 1000)
        )) {
            std::cerr << "WARNING: verbose mode is enabled and the inout matrices are large."
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
        const size_t size_temp = size_t(spec.n_rows_temp_) * size_t(spec.n_cols_temp_);
        const size_t size_A_bytes = size_A * sizeof(NumberA);
        const size_t size_temp_bytes = size_temp * sizeof(NumberTemp);
        const size_t inout_size_bytes = size_A_bytes;
        const size_t temp_size_bytes = size_temp_bytes;
        const size_t mem_size_bytes = inout_size_bytes + temp_size_bytes;
        constexpr float GB = 1024.0f * 1024.0f * 1024.0f;
        const float inout_size_gb = inout_size_bytes / GB;
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
            << "Inout matrix dimensions     : " << spec.n_rows_A_ << "x" << spec.n_cols_A_ << "\n"
            << "Temp matrix dimensions      : " << spec.n_rows_temp_ << "x" << spec.n_cols_temp_ << "\n"
            << "Inout size                  : " << inout_size_gb << " GB (" << inout_size_bytes << " bytes)\n"
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
        std::vector<NumberA> vec_A(size_A, 0);
        std::vector<NumberTemp> vec_temp(size_temp, 0);
        const auto setup_tp1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> setup_dt1 = setup_tp1 - setup_tp0;
        std::cout << setup_dt1.count() << " ms (" << setup_dt1.count() << " ms total)" << std::endl;

        std::cout << "  - Initializing matrices: ";
        if (is_random) {
            std::cout << "  (random) ";
            randomize_container(vec_A, seed);
        } else if (is_increasing) {
            std::cout << "  (increasing) ";
            for (size_t i = 0; i < size_A; ++i) vec_A[i] = NumberA(i);
        } else if (is_decreasing) {
            std::cout << "  (decreasing) ";
            for (size_t i = 0; i < size_A; ++i) vec_A[i] = NumberA(size_A - i);
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

        std::cout << "Matrix_Kernel_1Inout:" << std::endl;
        const auto gpu_tp0 = std::chrono::high_resolution_clock::now();
        cuda_check_error(cudaEventRecord(e0, stream), "cudaEventRecord");

        const auto gpu_step_1 = "Allocate device memory";
        NumberA* gpu_data_A = nullptr;
        NumberTemp* gpu_data_temp = nullptr;

        cuda_check_error(cudaMallocAsync(&gpu_data_A, size_A_bytes, stream), "cudaMallocAsync");
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
        kernel.run_device_kernel(gpu_data_A, gpu_data_temp, stream);
        cuda_check_error(cudaEventRecord(e3, stream), "cudaEventRecord");
        std::chrono::high_resolution_clock::time_point gpu_tp3{};
        cuda_check_error(cudaStreamAddCallback(stream, report_completion_time_callback, &gpu_tp3, NULL_FLAGS), "cudaStreamAddCallback");

        const auto gpu_step_4 = "Copy result back to host";
        auto vec_A_result_gpu = std::vector<NumberA>(size_A, 0);
        cuda_check_error(cudaMemcpyAsync(vec_A_result_gpu.data(), gpu_data_A, size_A_bytes, cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync");
        if (size_temp_bytes > 0) {
            cuda_check_error(cudaMemcpyAsync(vec_temp.data(), gpu_data_temp, size_temp_bytes, cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync");
        }
        cuda_check_error(cudaEventRecord(e4, stream), "cudaEventRecord");
        std::chrono::high_resolution_clock::time_point gpu_tp4{};
        cuda_check_error(cudaStreamAddCallback(stream, report_completion_time_callback, &gpu_tp4, NULL_FLAGS), "cudaStreamAddCallback");

        const auto gpu_step_5 = "Free device memory";
        cuda_check_error(cudaFreeAsync(gpu_data_A, stream), "cudaFreeAsync");
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
        const auto cpu_step_1 = "Convert data to Eigen, call copy constructor on A";
        auto vec_A_result_cpu{vec_A}; // Copy constructor
        const Eigen::Map<Eigen::Matrix<NumberA, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> A{vec_A.data(), spec.n_rows_A_, spec.n_cols_A_};
        const Eigen::Map<Eigen::Matrix<NumberA, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> A_result_gpu{vec_A_result_gpu.data(), spec.n_rows_A_, spec.n_cols_A_};
        Eigen::Map<Eigen::Matrix<NumberA, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> A_result_cpu{vec_A_result_cpu.data(), spec.n_rows_A_, spec.n_cols_A_};
        const auto cpu_tp1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> cpu_step_dt1 = cpu_tp1 - cpu_tp0;
        std::chrono::duration<double, std::milli> cpu_total_dt1 = cpu_tp1 - cpu_tp0;
        std::cout << " - " << std::setw(check_field_width) << cpu_step_1 << ": " << cpu_step_dt1.count() << " ms (" << cpu_total_dt1.count() << " ms total)" << std::endl;

        const auto cpu_step_2 = "Compute result with Eigen";
        kernel.run_host_kernel(A_result_cpu);
        static_assert(std::is_same_v<decltype(kernel.run_host_kernel(A_result_cpu)), void>);
        const auto cpu_tp2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> cpu_step_dt2 = cpu_tp2 - cpu_tp1;
        std::chrono::duration<double, std::milli> cpu_total_dt2 = cpu_tp2 - cpu_tp0;
        std::cout << " - " << std::setw(check_field_width) << cpu_step_2 << ": " << cpu_step_dt2.count() << " ms (" << cpu_total_dt2.count() << " ms total)" << std::endl;

        const auto cpu_step_3 = "Compute error matrix";
        const auto E = A_result_gpu.binaryExpr(A_result_cpu, compute_error_absolute).eval();
        const auto E_rel = A_result_gpu.binaryExpr(A_result_cpu, compute_error_relative).eval();
        const auto cpu_tp3 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> cpu_step_dt3 = cpu_tp3 - cpu_tp2;
        std::chrono::duration<double, std::milli> cpu_total_dt3 = cpu_tp3 - cpu_tp0;
        std::cout << " - " << std::setw(check_field_width) << cpu_step_3 << ": " << cpu_step_dt3.count() << " ms (" << cpu_total_dt3.count() << " ms total)" << std::endl;

        const auto cpu_step_4 = "Compute max error";
        size_t E_max_row, E_max_col, E_rel_max_row, E_rel_max_col;
        const double E_max = E.cwiseAbs().maxCoeff(&E_max_row, &E_max_col);
        const auto E_max_rel = E_rel.maxCoeff(&E_rel_max_row, &E_rel_max_col);
        const auto cpu_tp4 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> cpu_step_dt4 = cpu_tp4 - cpu_tp3;
        std::chrono::duration<double, std::milli> cpu_total_dt4 = cpu_tp4 - cpu_tp0;
        std::cout << " - " << std::setw(check_field_width) << cpu_step_4 << ": " << cpu_step_dt4.count() << " ms (" << cpu_total_dt4.count() << " ms total)" << std::endl;

        if (errors) {
            std::cout << "Non-zero error elements:\n";
            bool found_errors = false;
            for (long i = 0; i < E.rows(); ++i) {
                for (int j = 0; j < E.cols(); ++j) {
                    if (E(i, j) != NumberE(0)) {
                        found_errors = true;
                        std::cout << "(" << i << ", " << j << "): "
                                  << "result gpu=" << static_cast<Printable_Number<NumberA>>(A_result_gpu(i, j)) << ", "
                                  << "result cpu=" << static_cast<Printable_Number<NumberA>>(A_result_cpu(i, j)) << ", "
                                  << "E=" << static_cast<Printable_Number<NumberE>>(E(i, j)) << "\n";
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
            std::cout << A.template cast<Printable_Number<NumberA>>().format(eigen_format) << std::endl;
            std::cout << "A_gpu:\n";
            std::cout << A_result_gpu.template cast<Printable_Number<NumberA>>().format(eigen_format) << std::endl;
            std::cout << "A_cpu:\n";
            std::cout << A_result_cpu.template cast<Printable_Number<NumberA>>().format(eigen_format) << std::endl;
            if ((spec.n_rows_temp_ > 0) && (spec.n_cols_temp_ > 0)) {
                const Eigen::Map<Eigen::Matrix<NumberTemp, Eigen::Dynamic, 1>> tmp_gpu{vec_temp.data(), spec.n_temp_};
                std::cout << "tmp  :\n";
                std::cout << tmp_gpu.template cast<Printable_Number<NumberTemp>>().format(eigen_format) << std::endl;
            }
            std::cout << "E    :\n";
            std::cout << E.template cast<Printable_Number<NumberE>>().format(eigen_format) << std::endl;
            std::cout << "E_rel:\n";
            std::cout << E_rel.format(eigen_format) << std::endl;
        }

        const auto tp_done = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> total_dt = tp_done - setup_tp0;
        const auto tolerance = Tolerance::tolerance(tol_bits);
        std::cout << "DONE: " << total_dt.count() << " ms total" << std::endl;
        std::cout << "Max error     : " << E_max << " at (" << E_max_row << ", " << E_max_col << ")" << std::endl;
        std::cout << "Max error rel : " << E_max_rel << " at (" << E_rel_max_row << ", " << E_rel_max_col << ")" << std::endl;
        std::cout << "Precision rel : " << Tolerance::precision << " (" << Tolerance::name() << " with " << Tolerance::precision_bits << " bits of precision)" << std::endl;
        std::cout << "Tolerance rel : " << tolerance << " assuming a loss of " << tol_bits << " bits of precision" << std::endl;
        std::cout << "Gross speedup : " << (cpu_step_dt2.count()/gpu_step_dt3) << std::endl;
        std::cout << "Net speedup   : " << (cpu_total_dt2.count()/gpu_total_dt5) << std::endl;
        if (E_max == 0 || E_max_rel <= tolerance) {
            std::cout << "[SUCCESS]     : Max error pct is within tolerance" << std::endl;
        } else {
            std::cout << "[FAILURE]     : Max error pct exceeds tolerance" << std::endl;
        }
        return 0;
    }
};
