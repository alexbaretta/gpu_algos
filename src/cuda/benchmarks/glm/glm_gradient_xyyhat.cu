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


// source path: src/benchmarks/cuda/glm/glm_gradient_xyyhat.cu

#include <cxxopts.hpp>

#include "common/benchmark_options.hpp"
#include "cuda/benchmark/benchmark_tensor3d_3in_1out.cuh"

#include "cuda/kernels/glm/glm_gradient_xyyhat.cuh"

int main(int argc, char** argv) {
    cxxopts::Options options("glm_gradient_xyyhat", "Evaluate linear model");
    add_benchmark_options(options);
    Glm_gradient_xyyhat_spec::add_kernel_spec_options(options);

    try {
        cxxopts::ParseResult options_parsed = options.parse(argc, argv);

        Glm_gradient_xyyhat_spec spec = Glm_gradient_xyyhat_spec::make(options_parsed);

        if (spec.type_ == "half") {
            return Benchmark_Tensor3D_3In_1Out<Glm_gradient_xyyhat_kernel<__half>>(spec, options, options_parsed).run();
        } else if (spec.type_ == "single" || spec.type_ == "float") {
            return Benchmark_Tensor3D_3In_1Out<Glm_gradient_xyyhat_kernel<float>>(spec, options, options_parsed).run();
        } else if (spec.type_ == "double") {
            return Benchmark_Tensor3D_3In_1Out<Glm_gradient_xyyhat_kernel<double>>(spec, options, options_parsed).run();
        } else if (spec.type_ == "int8") {
            return Benchmark_Tensor3D_3In_1Out<Glm_gradient_xyyhat_kernel<std::int8_t>>(spec, options, options_parsed).run();
        } else if (spec.type_ == "int16") {
            return Benchmark_Tensor3D_3In_1Out<Glm_gradient_xyyhat_kernel<std::int16_t>>(spec, options, options_parsed).run();
        } else if (spec.type_ == "int32") {
            return Benchmark_Tensor3D_3In_1Out<Glm_gradient_xyyhat_kernel<std::int32_t>>(spec, options, options_parsed).run();
        } else if (spec.type_ == "int64") {
            return Benchmark_Tensor3D_3In_1Out<Glm_gradient_xyyhat_kernel<std::int64_t>>(spec, options, options_parsed).run();
        } else if (spec.type_ == "uint8") {
            return Benchmark_Tensor3D_3In_1Out<Glm_gradient_xyyhat_kernel<std::uint8_t>>(spec, options, options_parsed).run();
        } else if (spec.type_ == "uint16") {
            return Benchmark_Tensor3D_3In_1Out<Glm_gradient_xyyhat_kernel<std::uint16_t>>(spec, options, options_parsed).run();
        } else if (spec.type_ == "uint32") {
            return Benchmark_Tensor3D_3In_1Out<Glm_gradient_xyyhat_kernel<std::uint32_t>>(spec, options, options_parsed).run();
        } else if (spec.type_ == "uint64") {
            return Benchmark_Tensor3D_3In_1Out<Glm_gradient_xyyhat_kernel<std::uint64_t>>(spec, options, options_parsed).run();
        } else {
            throw cxxopts::exceptions::exception("Invalid type: " + spec.type_);
        }
    } catch (const cxxopts::exceptions::exception& e) {
       std::cerr << "Error parsing options: " << e.what() << std::endl;
       std::cout << options.help() << std::endl;
       return 1;
    }
}
