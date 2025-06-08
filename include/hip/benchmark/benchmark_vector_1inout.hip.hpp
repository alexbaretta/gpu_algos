// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

#pragma once

#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <Eigen/Dense>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cxxopts.hpp>

#include "common/random.hpp"
#include "hip/check_errors.hip.hpp"
#include "hip/hip_utils.hip.hpp"
#include "common/kernel_api/vector_1inout.hpp"

template <VECTOR_KERNEL_1INOUT Vector_kernel_1InOut>
class Benchmark_Vector_1InOut {
    public:
    using Kernel_spec = typename Vector_kernel_1InOut::Kernel_spec;
    using Number = typename Vector_kernel_1InOut::Number;
    using Printable_Number = std::conditional_t<std::is_same_v<Number, _Float16>, float, Number>;

    const Kernel_spec spec;
    const int seed;
    const int gpu_mem;
    const bool verbose;
    const bool errors;
    const bool force;
    const std::string init_method;

    Vector_kernel_1InOut kernel;

    template <typename... Args>
    Benchmark_Vector_1InOut(
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
        if (verbose && spec.n_A_ > 1000000) {
            std::cerr << "WARNING: verbose mode is enabled and the input vector is large."
            << "This will print the entire vector to the console." << std::endl;
            if (!force) {
                std::cerr << "Use --force to override." << std::endl
                          << "[ERROR] vector too big for verbose mode" << std::endl;
                exit(1);
            }
        }
    }

    int run() {
        const size_t size_A = size_t(spec.n_A_);
        const size_t size_temp = size_t(spec.n_temp_);
        const size_t size_A_bytes = size_A * sizeof(Number);
        const size_t size_temp_bytes = size_temp * sizeof(Number);
        const size_t input_size_bytes = size_A_bytes;
        const size_t temp_size_bytes = size_temp_bytes;
        const size_t mem_size_bytes = input_size_bytes + temp_size_bytes;
        constexpr float GB = 1024.0f * 1024.0f * 1024.0f;
        const float input_size_gb = input_size_bytes / GB;
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
            << "Input vector size           : " << spec.n_A_ << "\n"
            << "Temp vector size            : " << spec.n_temp_ << "\n"
            << "Input size                  : " << input_size_gb << " GB (" << input_size_bytes << " bytes)\n"
            << "Temp size                   : " << temp_size_gb << " GB (" << temp_size_bytes << " bytes)\n"
            << "Required memory             : " << mem_gb << " GB (" << mem_size_bytes << " bytes)\n"
            << std::endl;
        if (mem_gb > gpu_mem) {
            std::cerr << "[ERROR] GPU memory size is less than the vector size" << std::endl;
            return 1;
        }

        std::cout << "SETUP:" << std::endl;
        const auto setup_tp0 = std::chrono::high_resolution_clock::now();

        std::cout << "  - Allocating memory: ";
        std::vector<Number> vec_A(size_A, 0);
        std::vector<Number> vec_temp(size_temp, 0);
        const auto setup_tp1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> setup_dt1 = setup_tp1 - setup_tp0;
        std::cout << setup_dt1.count() << " ms (" << setup_dt1.count() << " ms total)" << std::endl;

        std::cout << "  - Initializing vector: ";
        if (is_random) {
            randomize_vector(vec_A, seed);
        } else if (is_increasing) {
            for (size_t i = 0; i < size_A; ++i) vec_A[i] = Number(i);
        } else if (is_decreasing) {
            for (size_t i = 0; i < size_A; ++i) vec_A[i] = Number(size_A - i);
        }
        const auto setup_tp2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> setup_step_dt2 = setup_tp2 - setup_tp1;
        std::chrono::duration<double, std::milli> setup_total_dt2 = setup_tp2 - setup_tp0;
        std::cout << setup_step_dt2.count() << " ms (" << setup_total_dt2.count() << " ms total)" << std::endl;

        std::cout << "  - Creating GPU streams: ";
        hipStream_t stream;
        hip_check_error(hipStreamCreate(&stream), "hipStreamCreate");
        const auto setup_tp3 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> setup_step_dt3 = setup_tp3 - setup_tp2;
        std::chrono::duration<double, std::milli> setup_total_dt3 = setup_tp3 - setup_tp0;
        std::cout << setup_step_dt3.count() << " ms (" << setup_total_dt3.count() << " ms total)" << std::endl;

        std::cout << "  - Creating GPU events: ";
        hipEvent_t e0, e1, e2, e3, e4, e5;
        hip_check_error(hipEventCreate(&e0), "hipEventCreate");
        hip_check_error(hipEventCreate(&e1), "hipEventCreate");
        hip_check_error(hipEventCreate(&e2), "hipEventCreate");
        hip_check_error(hipEventCreate(&e3), "hipEventCreate");
        hip_check_error(hipEventCreate(&e4), "hipEventCreate");
        hip_check_error(hipEventCreate(&e5), "hipEventCreate");
        const auto setup_tp4 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> setup_step_dt4 = setup_tp4 - setup_tp3;
        std::chrono::duration<double, std::milli> setup_total_dt4 = setup_tp4 - setup_tp0;
        std::cout << setup_step_dt4.count() << " ms (" << setup_total_dt4.count() << " ms total)" << std::endl;

        std::cout << "VECTOR_KERNEL_1INOUT:" << std::endl;
        const auto gpu_tp0 = std::chrono::high_resolution_clock::now();
        hip_check_error(hipEventRecord(e0, stream), "hipEventRecord");

        const auto gpu_step_1 = "Allocate device memory";
        Number *gpu_data_A = nullptr, *gpu_data_temp = nullptr;
        hip_check_error(hipMallocAsync(&gpu_data_A, size_A_bytes, stream), "hipMallocAsync");
        if (size_temp_bytes > 0) {
            hip_check_error(hipMallocAsync(&gpu_data_temp, size_temp_bytes, stream), "hipMallocAsync");
        }
        hip_check_error(hipEventRecord(e1, stream), "hipEventRecord");
        std::chrono::high_resolution_clock::time_point gpu_tp1{};
        hipStreamAddCallback(stream, report_completion_time_callback, &gpu_tp1, NULL_FLAGS);

        const auto gpu_step_2 = "Copy data to device";
        hip_check_error(hipMemcpyAsync(gpu_data_A, vec_A.data(), size_A_bytes, hipMemcpyHostToDevice, stream), "hipMemcpyAsync");
        hip_check_error(hipEventRecord(e2, stream), "hipEventRecord");
        std::chrono::high_resolution_clock::time_point gpu_tp2{};
        hipStreamAddCallback(stream, report_completion_time_callback, &gpu_tp2, NULL_FLAGS);

        const auto gpu_step_3 = "Compute kernel";
        kernel.run_device_kernel(gpu_data_A, gpu_data_temp, stream);
        hip_check_error(hipEventRecord(e3, stream), "hipEventRecord");
        std::chrono::high_resolution_clock::time_point gpu_tp3{};
        hip_check_error(hipStreamAddCallback(stream, report_completion_time_callback, &gpu_tp3, NULL_FLAGS), "hipStreamAddCallback");

        const auto gpu_step_4 = "Copy result back to host";
        hip_check_error(hipMemcpyAsync(vec_A.data(), gpu_data_A, size_A_bytes, hipMemcpyDeviceToHost, stream), "hipMemcpyAsync");
        if (size_temp_bytes > 0) {
            hip_check_error(hipMemcpyAsync(vec_temp.data(), gpu_data_temp, size_temp_bytes, hipMemcpyDeviceToHost, stream), "hipMemcpyAsync");
        }
        hip_check_error(hipEventRecord(e4, stream), "hipEventRecord");
        std::chrono::high_resolution_clock::time_point gpu_tp4{};
        hip_check_error(hipStreamAddCallback(stream, report_completion_time_callback, &gpu_tp4, NULL_FLAGS), "hipStreamAddCallback");

        const auto gpu_step_5 = "Free device memory";
        hip_check_error(hipFreeAsync(gpu_data_A, stream), "hipFreeAsync");
        if (size_temp_bytes > 0) {
            hip_check_error(hipFreeAsync(gpu_data_temp, stream), "hipFreeAsync");
        }
        hip_check_error(hipEventRecord(e5, stream), "hipEventRecord");
        std::chrono::high_resolution_clock::time_point gpu_tp5{};
        hip_check_error(hipStreamAddCallback(stream, report_completion_time_callback, &gpu_tp5, NULL_FLAGS), "hipStreamAddCallback");

        // Wait for stream to finish
        hip_check_error(hipStreamSynchronize(stream), "hipStreamSynchronize");

        // Print execution time
        constexpr int row_header_width = 22;
        constexpr int field_name_width = 25;
        float gpu_step_dt1 = 0.0f, gpu_step_dt2 = 0.0f, gpu_step_dt3 = 0.0f, gpu_step_dt4 = 0.0f, gpu_step_dt5 = 0.0f;
        float gpu_total_dt1 = 0.0f, gpu_total_dt2 = 0.0f, gpu_total_dt3 = 0.0f, gpu_total_dt4 = 0.0f, gpu_total_dt5 = 0.0f;

        std::chrono::duration<double, std::milli> chrono_step_dt1 = gpu_tp1 - gpu_tp0;
        std::chrono::duration<double, std::milli> chrono_total_dt1 = gpu_tp1 - gpu_tp0;
        hip_check_error(hipEventElapsedTime(&gpu_step_dt1, e0, e1), "hipEventElapsedTime");
        hip_check_error(hipEventElapsedTime(&gpu_total_dt1, e0, e1), "hipEventElapsedTime");
        std::cout << "1 - " << std::setw(row_header_width) << "std::chrono::duration " << std::setw(field_name_width) << gpu_step_1 << ": " << chrono_step_dt1.count() << " ms (" << chrono_total_dt1.count() << " ms total)" << std::endl;

        std::chrono::duration<double, std::milli> chrono_step_dt2 = gpu_tp2 - gpu_tp1;
        std::chrono::duration<double, std::milli> chrono_total_dt2 = gpu_tp2 - gpu_tp0;
        hip_check_error(hipEventElapsedTime(&gpu_step_dt2, e1, e2), "hipEventElapsedTime");
        hip_check_error(hipEventElapsedTime(&gpu_total_dt2, e0, e2), "hipEventElapsedTime");
        std::cout << "2 - " << std::setw(row_header_width) << "std::chrono::duration " << std::setw(field_name_width) << gpu_step_2 << ": " << chrono_step_dt2.count() << " ms (" << chrono_total_dt2.count() << " ms total)" << std::endl;

        std::chrono::duration<double, std::milli> chrono_step_dt3 = gpu_tp3 - gpu_tp2;
        std::chrono::duration<double, std::milli> chrono_total_dt3 = gpu_tp3 - gpu_tp0;
        hip_check_error(hipEventElapsedTime(&gpu_step_dt3, e2, e3), "hipEventElapsedTime");
        hip_check_error(hipEventElapsedTime(&gpu_total_dt3, e0, e3), "hipEventElapsedTime");
        std::cout << "3 - " << std::setw(row_header_width) << "std::chrono::duration " << std::setw(field_name_width) << gpu_step_3 << ": " << chrono_step_dt3.count() << " ms (" << chrono_total_dt3.count() << " ms total)" << std::endl;

        std::chrono::duration<double, std::milli> chrono_step_dt4 = gpu_tp4 - gpu_tp3;
        std::chrono::duration<double, std::milli> chrono_total_dt4 = gpu_tp4 - gpu_tp0;
        hip_check_error(hipEventElapsedTime(&gpu_step_dt4, e3, e4), "hipEventElapsedTime");
        hip_check_error(hipEventElapsedTime(&gpu_total_dt4, e0, e4), "hipEventElapsedTime");
        std::cout << "4 - " << std::setw(row_header_width) << "std::chrono::duration " << std::setw(field_name_width) << gpu_step_4 << ": " << chrono_step_dt4.count() << " ms (" << chrono_total_dt4.count() << " ms total)" << std::endl;

        std::chrono::duration<double, std::milli> chrono_step_dt5 = gpu_tp5 - gpu_tp4;
        std::chrono::duration<double, std::milli> chrono_total_dt5 = gpu_tp5 - gpu_tp0;
        hip_check_error(hipEventElapsedTime(&gpu_step_dt5, e4, e5), "hipEventElapsedTime");
        hip_check_error(hipEventElapsedTime(&gpu_total_dt5, e0, e5), "hipEventElapsedTime");
        std::cout << "5 - " << std::setw(row_header_width) << "std::chrono::duration " << std::setw(field_name_width) << gpu_step_5 << ": " << chrono_step_dt5.count() << " ms (" << chrono_total_dt5.count() << " ms total)" << std::endl;

        const auto cpu_tp0 = std::chrono::high_resolution_clock::now();

        constexpr int check_field_width = 26;
        std::cout << "CHECK WITH CPU:" << std::endl;
        const auto cpu_step_1 = "Convert data to Eigen";
        const Eigen::Map<Eigen::Matrix<Number, Eigen::Dynamic, 1>> A_cpu{vec_A.data(), spec.n_A_};
        const auto cpu_tp1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> cpu_step_dt1 = cpu_tp1 - cpu_tp0;
        std::chrono::duration<double, std::milli> cpu_total_dt1 = cpu_tp1 - cpu_tp0;
        std::cout << " - " << std::setw(check_field_width) << cpu_step_1 << ": " << cpu_step_dt1.count() << " ms (" << cpu_total_dt1.count() << " ms total)" << std::endl;

        const auto cpu_step_2 = "Store GPU result";
        const auto A_gpu = A_cpu.eval();
        const auto cpu_tp2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> cpu_step_dt2 = cpu_tp2 - cpu_tp1;
        std::chrono::duration<double, std::milli> cpu_total_dt2 = cpu_tp2 - cpu_tp0;
        std::cout << " - " << std::setw(check_field_width) << cpu_step_2 << ": " << cpu_step_dt2.count() << " ms (" << cpu_total_dt2.count() << " ms total)" << std::endl;

        const auto cpu_step_3 = "Compute result with Eigen";
        std::vector<Number> vec_A_original(size_A, 0);
        if (is_random) {
            randomize_vector(vec_A_original, seed);
        } else if (is_increasing) {
            for (size_t i = 0; i < size_A; ++i) vec_A_original[i] = Number(i);
        } else if (is_decreasing) {
            for (size_t i = 0; i < size_A; ++i) vec_A_original[i] = Number(size_A - i);
        }
        const Eigen::Map<Eigen::Matrix<Number, Eigen::Dynamic, 1>> A_original{vec_A_original.data(), spec.n_A_};
        const auto A_cpu_result = kernel.run_host_kernel(A_original);
        const auto cpu_tp3 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> cpu_step_dt3 = cpu_tp3 - cpu_tp2;
        std::chrono::duration<double, std::milli> cpu_total_dt3 = cpu_tp3 - cpu_tp0;
        std::cout << " - " << std::setw(check_field_width) << cpu_step_3 << ": " << cpu_step_dt3.count() << " ms (" << cpu_total_dt3.count() << " ms total)" << std::endl;

        const auto cpu_step_4 = "Compute error vector";
        const auto E = (A_gpu - A_cpu_result).eval();
        const auto E_pct = E.cwiseAbs().template cast<double>().array() / A_cpu_result.cwiseAbs().template cast<double>().array();
        const auto cpu_tp4 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> cpu_step_dt4 = cpu_tp4 - cpu_tp3;
        std::chrono::duration<double, std::milli> cpu_total_dt4 = cpu_tp4 - cpu_tp0;
        std::cout << " - " << std::setw(check_field_width) << cpu_step_4 << ": " << cpu_step_dt4.count() << " ms (" << cpu_total_dt4.count() << " ms total)" << std::endl;

        const auto cpu_step_5 = "Compute max error";
        size_t E_max_idx, E_pct_max_idx;
        const double E_max = E.cwiseAbs().maxCoeff(&E_max_idx);
        const auto E_max_pct = E_pct.maxCoeff(&E_pct_max_idx);
        const auto cpu_tp5 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> cpu_step_dt5 = cpu_tp5 - cpu_tp4;
        std::chrono::duration<double, std::milli> cpu_total_dt5 = cpu_tp5 - cpu_tp0;
        std::cout << " - " << std::setw(check_field_width) << cpu_step_5 << ": " << cpu_step_dt5.count() << " ms (" << cpu_total_dt5.count() << " ms total)" << std::endl;

        if (errors) {
            std::cout << "Non-zero error elements:\n";
            bool found_errors = false;
            for (int i = 0; i < E.size(); ++i) {
                if (E(i) != Number(0)) {
                    found_errors = true;
                    std::cout << "(" << i << "): "
                              << "A_original=" << static_cast<Printable_Number>(A_original(i)) << ", "
                              << "A_gpu=" << static_cast<Printable_Number>(A_gpu(i)) << ", "
                              << "A_cpu=" << static_cast<Printable_Number>(A_cpu_result(i)) << ", "
                              << "E=" << static_cast<Printable_Number>(E(i)) << "\n";
                }
            }
            if (!found_errors) {
                std::cout << "No non-zero error elements found.\n";
            }
        }

        if (verbose) {
            const Eigen::IOFormat clean_vector_format(4, 0, ", ", "\n", "  [", "]");
            std::cout << "A_original:\n";
            std::cout << A_original.template cast<Printable_Number>().format(clean_vector_format) << std::endl;
            std::cout << "A_gpu     :\n";
            std::cout << A_gpu.template cast<Printable_Number>().format(clean_vector_format) << std::endl;
            std::cout << "A_cpu     :\n";
            std::cout << A_cpu_result.template cast<Printable_Number>().format(clean_vector_format) << std::endl;
            if (spec.n_temp_ > 0) {
                const Eigen::Map<Eigen::Matrix<Number, Eigen::Dynamic, 1>> tmp_gpu{vec_temp.data(), spec.n_temp_};
                std::cout << "tmp       :\n";
                std::cout << tmp_gpu.template cast<Printable_Number>().format(clean_vector_format) << std::endl;
            }
        }

        const auto tp_done = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> total_dt = tp_done - setup_tp0;
        std::cout << "DONE: " << total_dt.count() << " ms total" << std::endl;
        std::cout << "Max error     : " << E_max << " at (" << E_max_idx << ")" << std::endl;
        std::cout << "Max error pct : " << E_max_pct << " at (" << E_pct_max_idx << ")" << std::endl;
        std::cout << "Gross speedup : " << (cpu_step_dt3.count()/gpu_step_dt3) << std::endl;
        std::cout << "Net speedup   : " << (cpu_total_dt3.count()/gpu_total_dt5) << std::endl;
        return 0;
    }
};
