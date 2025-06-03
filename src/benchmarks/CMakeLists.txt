# Copyright (c) 2025 Alessandro Baretta
# All rights reserved.

# source path: src/benchmarks/CMakeLists.txt

# Function to create a GPU executable with common settings
function(add_gpu_executable target_name source_file)
    add_executable(${target_name}
        ${source_file}
        ${CMAKE_SOURCE_DIR}/src/cuda/cuda_utils.cu
        ${CMAKE_SOURCE_DIR}/src/common/benchmark.cu
    )

    # Set CUDA specific properties
    set_target_properties(${target_name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES 89
    )

    # Link CUDA libraries including cuBLAS
    target_link_libraries(${target_name} PRIVATE
        cudart
        cuda
        CUDA::cublas
        CUDA::cublasLt
        Eigen3::Eigen
        OpenMP::OpenMP_CXX
    )
endfunction()

# Auto-discover and include all tool subdirectories
file(GLOB tool_dirs LIST_DIRECTORIES true RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *)
foreach(tool_dir ${tool_dirs})
    if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${tool_dir})
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${tool_dir}/CMakeLists.txt)
            message(STATUS "Including tool: ${tool_dir}")
            add_subdirectory(${tool_dir})
        endif()
    endif()
endforeach()
