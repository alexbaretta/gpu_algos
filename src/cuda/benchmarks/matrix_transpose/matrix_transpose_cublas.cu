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


// source path: src/cuda/benchmarks/matrix_transpose/matrix_transpose_cublas.cu

#include <cxxopts.hpp>

#include "cuda/benchmark/benchmark_matrix_1in_1out.cuh"
#include "common/benchmark_options.hpp"

#include "cuda/kernels/matrix_transpose/matrix_transpose_cublas.cuh"

int main(int argc, char** argv) {
    cxxopts::Options options("matrix_transpose_cublas", "Matrix transpose using cuBLAS");
    add_benchmark_options(options);
    Matrix_transpose_cublas_spec::add_kernel_spec_options(options);

    try {
        cxxopts::ParseResult options_parsed = options.parse(argc, argv);

        Matrix_transpose_cublas_spec spec = Matrix_transpose_cublas_spec::make(options_parsed);

        if (spec.type_ == "single" || spec.type_ == "float") {
            return Benchmark_Matrix_1In_1Out<Matrix_transpose_cublas_kernel<float>>(spec, options, options_parsed).run();
        } else if (spec.type_ == "double") {
            return Benchmark_Matrix_1In_1Out<Matrix_transpose_cublas_kernel<double>>(spec, options, options_parsed).run();
        } else {
            throw cxxopts::exceptions::exception("Invalid type: " + spec.type_);
        }
    } catch (const cxxopts::exceptions::exception& e) {
       std::cerr << "Error parsing options: " << e.what() << std::endl;
       std::cout << options.help() << std::endl;
       return 1;
    }
}
