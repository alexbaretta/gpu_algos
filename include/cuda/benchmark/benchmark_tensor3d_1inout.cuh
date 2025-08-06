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


// source path: include/cuda/benchmark/benchmark_tensor3d_1inout.cuh

#pragma once

#include <concepts>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <Eigen/Dense>

#include <cuda_runtime.h>
#include <cxxopts.hpp>

#include "common/types/tensor3d.hpp"
#include "cuda/check_errors.cuh"
#include "cuda/cuda_utils.cuh"
#include "cuda/kernel_api.cuh"

template <TENSOR3D_KERNEL_1INOUT Tensor3D_Kernel_1Inout>
class Benchmark_Tensor3D_1Inout {
    public:
    using Kernel_spec = typename Tensor3D_Kernel_1Inout::Kernel_spec;
    using Kernel = Tensor3D_Kernel_1Inout;
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

    Tensor3D_Kernel_1Inout kernel;

    template <typename... Args>
    Benchmark_Tensor3D_1Inout(
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
            (spec.n_rows_A_ > 10000 || spec.n_cols_A_ * spec.n_sheets_A_ > 1000)
        )) {
            std::cerr << "WARNING: verbose mode is enabled and the input tensors are large."
            << "This will print the entire tensors to the console." << std::endl;
            if (!force) {
                std::cerr << "Use --force to override." << std::endl
                          << "[ERROR] tensors too big for verbose mode" << std::endl;
                exit(1);
            }
        }
    }

    int run() {
        const size_t size_A = size_t(spec.n_rows_A_) * size_t(spec.n_cols_A_) * size_t(spec.n_sheets_A_);
        const size_t size_temp = size_t(spec.n_rows_temp_) * size_t(spec.n_cols_temp_) * size_t(spec.n_sheets_temp_);
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
            << "Inout tensor3d A dimensions : " << spec.n_rows_A_ << "x" << spec.n_cols_A_ << "x" << spec.n_sheets_A_ << "\n"
            << "Temp tensor3d dimensions    : " << spec.n_rows_temp_ << "x" << spec.n_cols_temp_ << "x" << spec.n_sheets_temp_ << "\n"
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
        Tensor3D<NumberA> tensor3d_A(spec.n_cols_A_, spec.n_rows_A_, spec.n_sheets_A_, 0);
        Tensor3D<NumberTemp> tensor3d_temp(spec.n_cols_temp_, spec.n_rows_temp_, spec.n_sheets_temp_, 0);
        const auto setup_tp1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> setup_dt1 = setup_tp1 - setup_tp0;
        std::cout << setup_dt1.count() << " ms (" << setup_dt1.count() << " ms total)" << std::endl;

        std::cout << "  - Initializing tensors: ";
        if (is_random) {
            std::cout << "  (random) ";
            tensor3d_A.randomize(seed);
        } else if (is_increasing) {
            std::cout << "  (increasing) ";
            for (size_t i = 0; i < size_A; ++i) tensor3d_A.vector_[i] = NumberA(i);
        } else if (is_increasing) {
            std::cout << "  (decreasing) ";
            for (size_t i = 0; i < size_A; ++i) tensor3d_A.vector_[i] = NumberA(size_A - i);
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

        std::cout << "Tensor3D_Kernel_1Inout:" << std::endl;
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
        cuda_check_error(cudaMemcpyAsync(gpu_data_A, tensor3d_A.data(), size_A_bytes, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync");
        cuda_check_error(cudaEventRecord(e2, stream), "cudaEventRecord");
        std::chrono::high_resolution_clock::time_point gpu_tp2{};
        cuda_check_error(cudaStreamAddCallback(stream, report_completion_time_callback, &gpu_tp2, NULL_FLAGS), "cudaStreamAddCallback");

        const auto gpu_step_3 = "Compute kernel";
        kernel.run_device_kernel(gpu_data_A, gpu_data_temp, stream);
        cuda_check_error(cudaEventRecord(e3, stream), "cudaEventRecord");
        std::chrono::high_resolution_clock::time_point gpu_tp3{};
        cuda_check_error(cudaStreamAddCallback(stream, report_completion_time_callback, &gpu_tp3, NULL_FLAGS), "cudaStreamAddCallback");

        const auto gpu_step_4 = "Copy result back to host";
        Tensor3D<NumberA> tensor3d_A_gpu(spec.n_cols_A_, spec.n_rows_A_, spec.n_sheets_A_, 0);
        cuda_check_error(cudaMemcpyAsync(tensor3d_A_gpu.data(), gpu_data_A, size_A_bytes, cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync");
        if (size_temp_bytes > 0) {
            cuda_check_error(cudaMemcpyAsync(tensor3d_temp.data(), gpu_data_temp, size_temp_bytes, cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync");
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
        const auto cpu_step_1 = "Convert data to Eigen (skipped for Tensor3D), call copy constructor on A";
        auto tensor3d_result_cpu{tensor3d_A}; // Copy constructor
        const auto cpu_tp1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> cpu_step_dt1 = cpu_tp1 - cpu_tp0;
        std::chrono::duration<double, std::milli> cpu_total_dt1 = cpu_tp1 - cpu_tp0;
        std::cout << " - " << std::setw(check_field_width) << cpu_step_1 << ": " << cpu_step_dt1.count() << " ms (" << cpu_total_dt1.count() << " ms total)" << std::endl;

        const auto cpu_step_2 = "Compute result with CPU";
        kernel.run_host_kernel(tensor3d_result_cpu);
        static_assert(std::is_same_v<decltype(kernel.run_host_kernel(tensor3d_result_cpu)), void>);
        const auto& tensor3d_result_gpu = tensor3d_A_gpu;
        const auto cpu_tp2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> cpu_step_dt2 = cpu_tp2 - cpu_tp1;
        std::chrono::duration<double, std::milli> cpu_total_dt2 = cpu_tp2 - cpu_tp0;
        std::cout << " - " << std::setw(check_field_width) << cpu_step_2 << ": " << cpu_step_dt2.count() << " ms (" << cpu_total_dt2.count() << " ms total)" << std::endl;

        const auto cpu_step_3 = "Compute error tensor3d and find max error";
        double E_max = 0;
        long E_max_row = 0, E_max_col = 0, E_max_sheet = 0;
        double E_max_rel = 0;
        long E_rel_max_row = 0, E_rel_max_col = 0, E_rel_max_sheet = 0;

        // row-major representation: innermost loop should iterate over elements of the same sheet/row
        const long e_rows = tensor3d_result_cpu.rows(), e_cols = tensor3d_result_cpu.cols(), e_sheets = tensor3d_result_cpu.sheets();
        Tensor3D<double> tensor3d_E(e_cols, e_rows, e_sheets, 0);
        Tensor3D<double> tensor3d_E_rel(e_cols, e_rows, e_sheets, 0);
        for (long sheet = 0; sheet < e_sheets; ++sheet) {
            for (long row = 0; row < e_rows; ++row) {
                for (long col = 0; col < e_cols; ++col) {
                    const double result_gpu_row_col = double(tensor3d_result_gpu(col, row, sheet));
                    const double result_cpu_row_col = double(tensor3d_result_cpu(col, row, sheet));
                    const double delta = result_gpu_row_col - result_cpu_row_col;
                    const bool results_are_identical = (
                        (std::isnan(result_gpu_row_col) && std::isnan(result_cpu_row_col))
                        || (std::isinf(result_gpu_row_col) && std::isinf(result_cpu_row_col) && std::isnan(delta))
                    );
                    const double e = results_are_identical ? 0 : delta;
                    const double e_abs = std::abs(e);
                    const double e_ref = double(tensor3d_result_cpu(col, row, sheet));
                    const double e_ref_abs = std::abs(e_ref);
                    const double e_rel = e_ref_abs > 0 ? e_abs / e_ref_abs : 0.0;
                    tensor3d_E(col, row, sheet) = e_abs;
                    tensor3d_E_rel(col, row, sheet) = e_rel;
                    if (e_abs > E_max) {
                        E_max = e_abs;
                        E_max_row = row;
                        E_max_col = col;
                        E_max_sheet = sheet;
                    }
                    if (e_rel > E_max_rel) {
                        E_max_rel = e_rel;
                        E_rel_max_row = row;
                        E_rel_max_col = col;
                        E_rel_max_sheet = sheet;
                    }
                }
            }
        }
        const auto cpu_tp3 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> cpu_step_dt3 = cpu_tp3 - cpu_tp2;
        std::chrono::duration<double, std::milli> cpu_total_dt3 = cpu_tp3 - cpu_tp0;
        std::cout << " - " << std::setw(check_field_width) << cpu_step_3 << ": " << cpu_step_dt3.count() << " ms (" << cpu_total_dt3.count() << " ms total)" << std::endl;

        if (errors) {
            std::cout << "Non-zero error elements:\n";
            bool found_errors = false;
            for (int i = 0; i < tensor3d_E.rows(); ++i) {
                for (int j = 0; j < tensor3d_E.cols(); ++j) {
                    for (int k = 0; k < tensor3d_E.sheets(); ++k) {
                        if (tensor3d_E(i, j, k) != 0.0) {
                            found_errors = true;
                            std::cout << "(" << i << ", " << j << ", " << k << "): "
                                    << "result gpu =" << static_cast<Printable_Number<NumberA>>(tensor3d_result_gpu(i, j, k)) << ", "
                                    << "result cpu =" << static_cast<Printable_Number<NumberA>>(tensor3d_result_cpu(i, j, k)) << ", "
                                    << "E          =" << static_cast<Printable_Number<NumberE>>(tensor3d_E(i, j, k)) << "\n";
                        }
                    }
                }
            }
            if (!found_errors) {
                std::cout << "No non-zero error elements found\n";
            }
        }

        if (verbose) {
            // const Eigen::IOFormat eigen_format(4, 0, ", ", "\n", "  [", "]");
            std::cout << "A    :\n";
            tensor3d_A.print(std::cout);
            std::cout << "A_gpu:\n";
            tensor3d_result_gpu.print(std::cout);
            std::cout << "A_cpu:\n";
            tensor3d_result_cpu.print(std::cout);
            if ((spec.n_rows_temp_ > 0) && (spec.n_cols_temp_ > 0) && (spec.n_sheets_temp_)) {
                std::cout << "tmp  :\n";
                tensor3d_temp.print(std::cout);
            }
            std::cout << "E    :\n";
            tensor3d_E.print(std::cout);
            std::cout << "E_rel:\n";
            tensor3d_E_rel.print(std::cout);
        }

        const auto tp_done = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> total_dt = tp_done - setup_tp0;
        const auto tolerance = Tolerance::tolerance(tol_bits);
        std::cout << "DONE: " << total_dt.count() << " ms total" << std::endl;
        std::cout << "Max error     : " << E_max << " at (" << E_max_row << ", " << E_max_col << ", " << E_max_sheet << ")" << std::endl;
        std::cout << "Max error rel : " << E_max_rel << " at (" << E_rel_max_row << ", " << E_rel_max_col << ", " << E_rel_max_sheet << ")" << std::endl;
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
