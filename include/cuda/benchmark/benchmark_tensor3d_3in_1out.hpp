// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: include/cuda/benchmark/benchmark_tensor3d_3in_1out.hpp

#pragma once

#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <Eigen/Dense>

#include <cuda_runtime.h>
#include <cxxopts.hpp>

#include "common/types/tensor3d.hpp"
#include "common/random.hpp"
#include "cuda/check_errors.hpp"
#include "cuda/cuda_utils.hpp"
#include "common/kernel_api/tensor3d_3in_1out.hpp"

template <TENSOR3D_KERNEL_3IN_1OUT Tensor3d_kernel_3In_1Out>
class Benchmark_Tensor3D_3In_1Out {
    public:
    using Kernel_spec = typename Tensor3d_kernel_3In_1Out::Kernel_spec;
    using Number = typename Tensor3d_kernel_3In_1Out::Number;
    using Printable_Number = std::conditional_t<std::is_same_v<Number, __half>, float, Number>;

    const Kernel_spec spec;
    const int seed;
    const int gpu_mem;
    const bool verbose;
    const bool errors;
    const bool force;
    const std::string init_method;

    Tensor3d_kernel_3In_1Out kernel;

    template <typename... Args>
    Benchmark_Tensor3D_3In_1Out(
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
        const long total_elements_A = spec.n_rows_A_ * spec.n_cols_A_ * spec.n_sheets_A_;
        const long total_elements_B = spec.n_rows_B_ * spec.n_cols_B_ * spec.n_sheets_B_;
        const long total_elements_C = spec.n_rows_C_ * spec.n_cols_C_ * spec.n_sheets_C_;
        const long max_elements = std::max({total_elements_A, total_elements_B, total_elements_C});
        if (verbose && max_elements > 10000) {
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
        const size_t size_B = size_t(spec.n_rows_B_) * size_t(spec.n_cols_B_) * size_t(spec.n_sheets_B_);
        const size_t size_C = size_t(spec.n_rows_C_) * size_t(spec.n_cols_C_) * size_t(spec.n_sheets_C_);
        const size_t size_D = size_t(spec.n_rows_D_) * size_t(spec.n_cols_D_) * size_t(spec.n_sheets_D_);
        const size_t size_temp = size_t(spec.n_rows_temp_) * size_t(spec.n_cols_temp_) * size_t(spec.n_sheets_temp_);
        const size_t size_A_bytes = size_A * sizeof(Number);
        const size_t size_B_bytes = size_B * sizeof(Number);
        const size_t size_C_bytes = size_C * sizeof(Number);
        const size_t size_D_bytes = size_D * sizeof(Number);
        const size_t size_temp_bytes = size_temp * sizeof(Number);
        const size_t input_size_bytes = size_A_bytes + size_B_bytes + size_C_bytes;
        const size_t output_size_bytes = size_D_bytes;
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
            << "Input tensor A dimensions    : " << spec.n_rows_A_ << "x" << spec.n_cols_A_ << "x" << spec.n_sheets_A_ << "\n"
            << "Input tensor B dimensions    : " << spec.n_rows_B_ << "x" << spec.n_cols_B_ << "x" << spec.n_sheets_B_ << "\n"
            << "Input tensor C dimensions    : " << spec.n_rows_C_ << "x" << spec.n_cols_C_ << "x" << spec.n_sheets_C_ << "\n"
            << "Output tensor dimensions     : " << spec.n_rows_D_ << "x" << spec.n_cols_D_ << "x" << spec.n_sheets_D_ << "\n"
            << "Temp tensor dimensions       : " << spec.n_rows_temp_ << "x" << spec.n_cols_temp_ << "x" << spec.n_sheets_temp_ << "\n"
            << "Input size                   : " << input_size_gb << " GB (" << input_size_bytes << " bytes)\n"
            << "Output size                  : " << output_size_gb << " GB (" << output_size_bytes << " bytes)\n"
            << "Temp size                    : " << temp_size_gb << " GB (" << temp_size_bytes << " bytes)\n"
            << "Required memory              : " << mem_gb << " GB (" << mem_size_bytes << " bytes)\n"
            << std::endl;
        if (mem_gb > gpu_mem) {
            std::cerr << "[ERROR] GPU memory size is less than the tensor size" << std::endl;
            return 1;
        }

        std::cout << "SETUP:" << std::endl;
        const auto setup_tp0 = std::chrono::high_resolution_clock::now();

        std::cout << "  - Allocating memory: ";
        Tensor3D<Number> A(spec.n_rows_A_, spec.n_cols_A_, spec.n_sheets_A_, 0);
        Tensor3D<Number> B(spec.n_rows_B_, spec.n_cols_B_, spec.n_sheets_B_, 0);
        Tensor3D<Number> C(spec.n_rows_C_, spec.n_cols_C_, spec.n_sheets_C_, 0);
        Tensor3D<Number> D(spec.n_rows_D_, spec.n_cols_D_, spec.n_sheets_D_, 0);
        Tensor3D<Number> temp(spec.n_rows_temp_, spec.n_cols_temp_, spec.n_sheets_temp_, 0);
        const auto setup_tp1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> setup_dt1 = setup_tp1 - setup_tp0;
        std::cout << setup_dt1.count() << " ms (" << setup_dt1.count() << " ms total)" << std::endl;

        std::cout << "  - Initializing tensors: ";
        if (is_random) {
            randomize_tensor(A, seed);
            randomize_tensor(B, seed+1);
            randomize_tensor(C, seed+2);
        } else if (is_increasing) {
            for (size_t i = 0; i < size_A; ++i) A.data()[i] = Number(i);
            for (size_t i = 0; i < size_B; ++i) B.data()[i] = Number(i);
            for (size_t i = 0; i < size_C; ++i) C.data()[i] = Number(i);
        } else if (is_decreasing) {
            for (size_t i = 0; i < size_A; ++i) A.data()[i] = Number(size_A - i);
            for (size_t i = 0; i < size_B; ++i) B.data()[i] = Number(size_B - i);
            for (size_t i = 0; i < size_C; ++i) C.data()[i] = Number(size_C - i);
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

        std::cout << "TENSOR3D_KERNEL_3IN_1OUT:" << std::endl;
        const auto gpu_tp0 = std::chrono::high_resolution_clock::now();
        cuda_check_error(cudaEventRecord(e0, stream), "cudaEventRecord");

        const auto gpu_step_1 = "Allocate device memory";
        Number *gpu_data_A = nullptr, *gpu_data_B = nullptr, *gpu_data_C = nullptr, *gpu_data_D = nullptr, *gpu_data_temp = nullptr;
        cuda_check_error(cudaMallocAsync(&gpu_data_A, size_A_bytes, stream), "cudaMallocAsync");
        cuda_check_error(cudaMallocAsync(&gpu_data_B, size_B_bytes, stream), "cudaMallocAsync");
        cuda_check_error(cudaMallocAsync(&gpu_data_C, size_C_bytes, stream), "cudaMallocAsync");
        cuda_check_error(cudaMallocAsync(&gpu_data_D, size_D_bytes, stream), "cudaMallocAsync");
        if (size_temp_bytes > 0) {
            cuda_check_error(cudaMallocAsync(&gpu_data_temp, size_temp_bytes, stream), "cudaMallocAsync");
        }
        cuda_check_error(cudaEventRecord(e1, stream), "cudaEventRecord");
        std::chrono::high_resolution_clock::time_point gpu_tp1{};
        cudaStreamAddCallback(stream, report_completion_time_callback, &gpu_tp1, NULL_FLAGS);

        const auto gpu_step_2 = "Copy data to device";
        cuda_check_error(cudaMemcpyAsync(gpu_data_A, A.data(), size_A_bytes, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync");
        cuda_check_error(cudaMemcpyAsync(gpu_data_B, B.data(), size_B_bytes, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync");
        cuda_check_error(cudaMemcpyAsync(gpu_data_C, C.data(), size_C_bytes, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync");
        cuda_check_error(cudaEventRecord(e2, stream), "cudaEventRecord");
        std::chrono::high_resolution_clock::time_point gpu_tp2{};
        cudaStreamAddCallback(stream, report_completion_time_callback, &gpu_tp2, NULL_FLAGS);

        const auto gpu_step_3 = "Compute kernel";
        kernel.run_device_kernel(gpu_data_A, gpu_data_B, gpu_data_C, gpu_data_D, gpu_data_temp, stream);
        cuda_check_error(cudaEventRecord(e3, stream), "cudaEventRecord");
        std::chrono::high_resolution_clock::time_point gpu_tp3{};
        cuda_check_error(cudaStreamAddCallback(stream, report_completion_time_callback, &gpu_tp3, NULL_FLAGS), "cudaStreamAddCallback");

        const auto gpu_step_4 = "Copy result back to host";
        cuda_check_error(cudaMemcpyAsync(D.data(), gpu_data_D, size_D_bytes, cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync");
        if (size_temp_bytes > 0) {
            cuda_check_error(cudaMemcpyAsync(temp.data(), gpu_data_temp, size_temp_bytes, cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync");
        }
        cuda_check_error(cudaEventRecord(e4, stream), "cudaEventRecord");
        std::chrono::high_resolution_clock::time_point gpu_tp4{};
        cuda_check_error(cudaStreamAddCallback(stream, report_completion_time_callback, &gpu_tp4, NULL_FLAGS), "cudaStreamAddCallback");

        const auto gpu_step_5 = "Free device memory";
        cuda_check_error(cudaFreeAsync(gpu_data_A, stream), "cudaFreeAsync");
        cuda_check_error(cudaFreeAsync(gpu_data_B, stream), "cudaFreeAsync");
        cuda_check_error(cudaFreeAsync(gpu_data_C, stream), "cudaFreeAsync");
        cuda_check_error(cudaFreeAsync(gpu_data_D, stream), "cudaFreeAsync");
        if (size_temp_bytes > 0) {
            cuda_check_error(cudaFreeAsync(gpu_data_temp, stream), "cudaFreeAsync");
        }
        cuda_check_error(cudaEventRecord(e5, stream), "cudaEventRecord");
        std::chrono::high_resolution_clock::time_point gpu_tp5{};
        cuda_check_error(cudaStreamAddCallback(stream, report_completion_time_callback, &gpu_tp5, NULL_FLAGS), "cudaStreamAddCallback");

        cuda_check_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

        float gpu_step_dt1, gpu_step_dt2, gpu_step_dt3, gpu_step_dt4, gpu_step_dt5;
        cuda_check_error(cudaEventElapsedTime(&gpu_step_dt1, e0, e1), "cudaEventElapsedTime");
        cuda_check_error(cudaEventElapsedTime(&gpu_step_dt2, e1, e2), "cudaEventElapsedTime");
        cuda_check_error(cudaEventElapsedTime(&gpu_step_dt3, e2, e3), "cudaEventElapsedTime");
        cuda_check_error(cudaEventElapsedTime(&gpu_step_dt4, e3, e4), "cudaEventElapsedTime");
        cuda_check_error(cudaEventElapsedTime(&gpu_step_dt5, e4, e5), "cudaEventElapsedTime");
        const float gpu_total_dt5 = gpu_step_dt1 + gpu_step_dt2 + gpu_step_dt3 + gpu_step_dt4 + gpu_step_dt5;

        std::cout << "  - " << gpu_step_1 << ": " << gpu_step_dt1 << " ms" << std::endl;
        std::cout << "  - " << gpu_step_2 << ": " << gpu_step_dt2 << " ms" << std::endl;
        std::cout << "  - " << gpu_step_3 << ": " << gpu_step_dt3 << " ms" << std::endl;
        std::cout << "  - " << gpu_step_4 << ": " << gpu_step_dt4 << " ms" << std::endl;
        std::cout << "  - " << gpu_step_5 << ": " << gpu_step_dt5 << " ms" << std::endl;
        std::cout << "  - Total GPU time: " << gpu_total_dt5 << " ms" << std::endl;

        std::cout << "CPU:" << std::endl;
        const auto cpu_tp0 = std::chrono::high_resolution_clock::now();

        std::cout << "  - Computing reference: ";
        const auto D_cpu = kernel.run_host_kernel(A, B, C);
        const auto cpu_tp1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> cpu_step_dt1 = cpu_tp1 - cpu_tp0;
        std::chrono::duration<double, std::milli> cpu_total_dt1 = cpu_tp1 - cpu_tp0;
        std::cout << cpu_step_dt1.count() << " ms (" << cpu_total_dt1.count() << " ms total)" << std::endl;

        std::cout << "  - Computing error: ";
        Tensor3D<Number> E(spec.n_rows_D_, spec.n_cols_D_, spec.n_sheets_D_, 0);
        for (size_t i = 0; i < size_D; ++i) {
            E.data()[i] = D.data()[i] - D_cpu.data()[i];
        }
        const auto cpu_tp2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> cpu_step_dt2 = cpu_tp2 - cpu_tp1;
        std::chrono::duration<double, std::milli> cpu_total_dt2 = cpu_tp2 - cpu_tp0;
        std::cout << cpu_step_dt2.count() << " ms (" << cpu_total_dt2.count() << " ms total)" << std::endl;

        std::cout << "  - Computing max error: ";
        double E_max = 0;
        long E_max_row = 0, E_max_col = 0, E_max_sheet = 0;
        double E_max_pct = 0;
        long E_pct_max_row = 0, E_pct_max_col = 0, E_pct_max_sheet = 0;
        for (long sheet = 0; sheet < spec.n_sheets_D_; ++sheet) {
            for (long row = 0; row < spec.n_rows_D_; ++row) {
                for (long col = 0; col < spec.n_cols_D_; ++col) {
                    const double e = double(E(row, col, sheet));
                    const double e_abs = std::abs(e);
                    const double e_ref = double(D_cpu(row, col, sheet));
                    const double e_ref_abs = std::abs(e_ref);
                    const double e_pct = e_ref_abs > 0 ? 100.0 * e_abs / e_ref_abs : 0.0;
                    if (e_abs > E_max) {
                        E_max = e_abs;
                        E_max_row = row;
                        E_max_col = col;
                        E_max_sheet = sheet;
                    }
                    if (e_pct > E_max_pct) {
                        E_max_pct = e_pct;
                        E_pct_max_row = row;
                        E_pct_max_col = col;
                        E_pct_max_sheet = sheet;
                    }
                }
            }
        }
        const auto cpu_tp3 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> cpu_step_dt3 = cpu_tp3 - cpu_tp2;
        std::chrono::duration<double, std::milli> cpu_total_dt3 = cpu_tp3 - cpu_tp0;
        std::cout << cpu_step_dt3.count() << " ms (" << cpu_total_dt3.count() << " ms total)" << std::endl;

        if (verbose) {
            std::cout << "A      :\n" << A << std::endl;
            std::cout << "B      :\n" << B << std::endl;
            std::cout << "C      :\n" << C << std::endl;
            std::cout << "D_gpu  :\n" << D << std::endl;
            std::cout << "D_cpu  :\n" << D_cpu << std::endl;
            if (spec.n_rows_temp_ > 0 && spec.n_cols_temp_ > 0 && spec.n_sheets_temp_ > 0) {
                std::cout << "temp   :\n" << temp << std::endl;
            }
        }

        const auto tp_done = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> total_dt = tp_done - setup_tp0;
        std::cout << "DONE: " << total_dt.count() << " ms total" << std::endl;
        std::cout << "Max error: " << E_max << " at (" << E_max_row << ", " << E_max_col << ", " << E_max_sheet << ")" << std::endl;
        std::cout << "Max error percentage: " << E_max_pct << " at (" << E_pct_max_row << ", " << E_pct_max_col << ", " << E_pct_max_sheet << ")" << std::endl;
        std::cout << "Gross speedup : " << (cpu_step_dt1.count()/gpu_step_dt3) << std::endl;
        std::cout << "Net speedup   : " << (cpu_total_dt1.count()/gpu_total_dt5) << std::endl;

        // Clean up
        cuda_check_error(cudaEventDestroy(e0), "cudaEventDestroy");
        cuda_check_error(cudaEventDestroy(e1), "cudaEventDestroy");
        cuda_check_error(cudaEventDestroy(e2), "cudaEventDestroy");
        cuda_check_error(cudaEventDestroy(e3), "cudaEventDestroy");
        cuda_check_error(cudaEventDestroy(e4), "cudaEventDestroy");
        cuda_check_error(cudaEventDestroy(e5), "cudaEventDestroy");
        cuda_check_error(cudaStreamDestroy(stream), "cudaStreamDestroy");

        return 0;
    }
};
