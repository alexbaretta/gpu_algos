# Copyright (c) 2025 Alessandro Baretta
# All rights reserved.

cmake_minimum_required(VERSION 4.0)  # Lowered requirement for hip::host usage

# Set C++ compiler before project()
set(CMAKE_CXX_COMPILER "g++-14")
set(CMAKE_CUDA_HOST_COMPILER "g++-14")
set(CMAKE_CUDA_COMPILER "nvcc")
set(CMAKE_HIP_COMPILER "/opt/rocm/bin/amdclang++")
set(CMAKE_HIP_HOST_COMPILER "/opt/rocm/bin/amdclang++")
set(GPU_TARGETS "gfx1201")  # AMD Radeon RX 9070 (RDNA 3+)

project(gpu_algos LANGUAGES CUDA CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set default parallel jobs for building
set(CMAKE_BUILD_PARALLEL_LEVEL 12)

# Create meta-targets for different technology stacks
add_custom_target(cuda-targets
    COMMENT "Building all CUDA targets"
)

add_custom_target(hip-targets
    COMMENT "Building all HIP targets"
)

add_custom_target(cpp-targets
    COMMENT "Building all C++ targets"
)

# Set CUDA specific flags
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "89")  # Compute capability 8.9


find_package(CUDAToolkit REQUIRED)


# Try to enable HIP if available
include(CheckLanguage)
check_language(HIP)
message(STATUS "HIP compiler found: ${CMAKE_HIP_COMPILER}")


execute_process(
    COMMAND bash -c "${CMAKE_CUDA_HOST_COMPILER} --print-search-dirs | grep libraries | cut -d= -f2 | tr ':' ' ' | xargs readlink -f | tr '\n' ';' | head -c -1"
    OUTPUT_VARIABLE CUDA_HOST_COMPILER_LIBRARY_PATH
    COMMAND_ECHO STDOUT
    COMMAND_ERROR_IS_FATAL ANY
)

if(CMAKE_HIP_COMPILER)
    set(CMAKE_HIP_STANDARD 20)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)
    set(CMAKE_HIP_ARCHITECTURES "${GPU_TARGETS}")  # AMD Radeon RX 9070 (RDNA 3+)

    # Enable HIP language
    enable_language(HIP)

    execute_process(
        COMMAND bash -c "${CMAKE_HIP_HOST_COMPILER} --print-search-dirs | grep libraries | cut -d= -f2 | tr ':' ' ' | xargs readlink -f | tr '\n' ';' | head -c -1"
        OUTPUT_VARIABLE HIP_HOST_COMPILER_LIBRARY_PATH
        COMMAND_ECHO STDOUT
        COMMAND_ERROR_IS_FATAL ANY
    )

    # Set language mappings for HIP files
    set_property(GLOBAL PROPERTY CMAKE_HIP_LANGUAGE_EXTENSION ".hip")

    # Add ROCm cmake path so find_package can locate HIP
    list(APPEND CMAKE_PREFIX_PATH "/opt/rocm")

    # Find HIP package using the modern CMake approach
    find_package(hip CONFIG QUIET)
    if(hip_FOUND)
        message(STATUS "HIP found and enabled")
        # Set HIP specific flags
        set(HIP_AVAILABLE TRUE)

        # Find additional ROCm libraries that HIP targets may need
        find_package(rocblas CONFIG QUIET)
        if(rocblas_FOUND)
            message(STATUS "rocBLAS found")
        endif()

        # Add OpenMP flags for HIP
        set(DEFAULT_HIP_FLAGS "${DEFAULT_HIP_FLAGS} -fopenmp")

    else()
        message(WARNING "HIP compiler found but HIP package not found - HIP support disabled")
        set(HIP_AVAILABLE FALSE)
    endif()
else()
    message(STATUS "HIP compiler not found - HIP support disabled")
    set(HIP_AVAILABLE FALSE)
endif()

# Set output directory for executables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Force override CUDAToolkit's librt detection to prevent linking with it
# (librt is unnecessary in modern glibc and causes nvlink issues)
# Clear any existing cached rt library references
# unset(CUDAToolkit_rt_LIBRARY CACHE)
# set(CUDAToolkit_rt_LIBRARY "" CACHE FILEPATH "Force disable librt linking" FORCE)
# set(CUDAToolkit_rt_LIBRARY_RELEASE "" CACHE FILEPATH "Force disable librt linking (Release)" FORCE)
# set(CUDAToolkit_rt_LIBRARY_DEBUG "" CACHE FILEPATH "Force disable librt linking (Debug)" FORCE)


# Force override CUDAToolkit's librt detection to prevent linking with it
# (librt is unnecessary in modern glibc and causes nvlink issues)
# set(CUDAToolkit_rt_LIBRARY "" CACHE FILEPATH "Force disable librt linking" FORCE)

# Find OpenMP and let CMake automatically detect the correct implementation
find_package(OpenMP REQUIRED)

# Add OpenMP flags for CUDA (avoid duplication)
set(REQUIRED_CUDA_FLAGS "${REQUIRED_CUDA_FLAGS} -Xcompiler=-fopenmp,-Wall,-Werror -diag-suppress 20012,20013 --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${REQUIRED_CUDA_FLAGS}")
set(DEFAULT_CUDA_FLAGS "${DEFAULT_CUDA_FLAGS}")

# We need to disable GPU support in Eigen in HIP mode because Eigen uses 4-byte masks,
# which in CUDA is allowed but in HIP fails.
set(DEFAULT_HIP_FLAGS "${DEFAULT_HIP_FLAGS} -fopenmp -xhip -DHIP_ENABLE_WARP_SYNC_BUILTINS -DEIGEN_NO_CUDA")
set(DEFAULT_LINK_FLAGS "${DEFAULT_LINK_FLAGS} -fopenmp")


# Link OpenMP to all targets
link_libraries(OpenMP::OpenMP_CXX)

# Add compiler warnings (avoid duplication by checking if already set)
if(NOT CMAKE_CXX_FLAGS MATCHES "-Werror")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall")
endif()
if(NOT DEFAULT_CUDA_FLAGS MATCHES "--Werror")
    set(DEFAULT_CUDA_FLAGS "${DEFAULT_CUDA_FLAGS} --Werror cross-execution-space-call,reorder,ext-lambda-captures-this,deprecated-declarations")
endif()
if(HIP_AVAILABLE AND NOT DEFAULT_HIP_FLAGS MATCHES "-Werror")
    set(DEFAULT_HIP_FLAGS "${DEFAULT_HIP_FLAGS} -Werror -Wall")
endif()

# Find Eigen package
find_package(Eigen3 REQUIRED)

# Add include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    # ${CMAKE_CURRENT_SOURCE_DIR}/include/cuda
    ${CMAKE_CURRENT_SOURCE_DIR}/third-party/cxxopts/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third-party/cutlass/include
    ${EIGEN3_INCLUDE_DIR}
)

# Conditionally add HIP include directory
if(HIP_AVAILABLE)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/hip)
endif()

# Function to create a GPU executable with common settings
function(add_gpu_executable target_name source_file)
    add_executable(${target_name}
        ${source_file}
        ${CMAKE_SOURCE_DIR}/src/cuda/cuda_utils.cu
        ${CMAKE_SOURCE_DIR}/src/common/benchmark_options.cpp
    )

    separate_arguments(LOCAL_CUDA_FLAGS UNIX_COMMAND ${DEFAULT_CUDA_FLAGS})
    separate_arguments(LOCAL_LINK_FLAGS UNIX_COMMAND ${DEFAULT_LINK_FLAGS})
    target_link_options(${target_name} PRIVATE "${LOCAL_LINK_FLAGS}")
    set_source_files_properties(${source_file} PROPERTIES
        COMPILE_OPTIONS "${LOCAL_CUDA_FLAGS}"
        LANGUAGE CUDA
    )

    message(STATUS "COMPILE_OPTIONS for CUDA ${source_file}: ${DEFAULT_CUDA_FLAGS}")

    # Set CUDA specific properties
    set_target_properties(${target_name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RUNTIME_LIBRARY Shared
        CUDA_ARCHITECTURES 89
        LINKER_LANGUAGE CUDA
    )

    # Add include directories
    target_include_directories(${target_name} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/cuda
        ${CMAKE_SOURCE_DIR}/third-party/cxxopts/include
        ${EIGEN3_INCLUDE_DIR}
    )

    # Link CUDA libraries
    target_link_directories(
        ${target_name} PRIVATE
        ${CUDAToolkit_LIBRARY_DIR}
        ${CUDA_HOST_COMPILER_LIBRARY_PATH}
    )

    target_link_libraries(${target_name} PRIVATE
        CUDA::cudart
        CUDA::cuda_driver
        CUDA::cublas
        CUDA::cublasLt
        Eigen3::Eigen
        OpenMP::OpenMP_CXX
    )

    # Add this target to the cuda-targets meta-target
    add_dependencies(cuda-targets ${target_name})
endfunction()

# Function to create a HIP executable (only available when HIP is enabled)
function(add_hip_executable target_name source_file)
    if(NOT HIP_AVAILABLE)
        message(STATUS "Skipping HIP target ${target_name} - HIP not available")
        return()
    endif()

    add_executable(${target_name}
        ${source_file}
        ${CMAKE_SOURCE_DIR}/src/hip/hip_utils.hip
        ${CMAKE_SOURCE_DIR}/src/common/benchmark_options.cpp
        ${CMAKE_SOURCE_DIR}/src/hip/benchmark.hip
    )

    separate_arguments(LOCAL_HIP_FLAGS UNIX_COMMAND ${DEFAULT_HIP_FLAGS})
    separate_arguments(LOCAL_LINK_FLAGS UNIX_COMMAND ${DEFAULT_LINK_FLAGS})
    target_link_options(${target_name} PRIVATE "${LOCAL_LINK_FLAGS}")
    set_source_files_properties(${source_file} PROPERTIES
        COMPILE_OPTIONS "${LOCAL_HIP_FLAGS}"
        LANGUAGE HIP
    )

    message(STATUS "COMPILE_OPTIONS for HIP ${source_file}: ${DEFAULT_HIP_FLAGS}")


    # Set the language for HIP source files explicitly
    set_source_files_properties(${CMAKE_SOURCE_DIR}/src/hip/hip_utils.hip PROPERTIES LANGUAGE HIP)
    set_source_files_properties(${CMAKE_SOURCE_DIR}/src/hip/benchmark.hip PROPERTIES LANGUAGE HIP)

    # Set HIP specific properties
    set_target_properties(${target_name} PROPERTIES
        HIP_STANDARD 20
        HIP_ARCHITECTURES "gfx1201"
    )

    # Add include directories
    target_include_directories(${target_name} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/hip
        ${CMAKE_SOURCE_DIR}/third-party/cxxopts/include
        ${EIGEN3_INCLUDE_DIR}
    )

    # Link HIP libraries
    target_link_directories(
        ${target_name} PRIVATE
        ${HIP_HOST_COMPILER_LIBRARY_PATH}
    )

    target_link_libraries(${target_name} PRIVATE
        hip::host
        Eigen3::Eigen
        OpenMP::OpenMP_CXX
    )

    # Add this target to the hip-targets meta-target
    add_dependencies(hip-targets ${target_name})
endfunction()

# Function to create a C++ executable
function(add_cpp_executable target_name source_file)
    add_executable(${target_name}
        ${source_file}
        ${CMAKE_SOURCE_DIR}/src/common/benchmark_options.cpp
    )

    # Set C++ specific properties
    set_target_properties(${target_name} PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
    )

    # Add include directories
    target_include_directories(${target_name} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/common
        ${CMAKE_SOURCE_DIR}/third-party/cxxopts/include
        ${EIGEN3_INCLUDE_DIR}
    )

    # Link libraries
    target_link_libraries(${target_name} PRIVATE
        Eigen3::Eigen
        OpenMP::OpenMP_CXX
    )

    # Add this target to the cpp-targets meta-target
    add_dependencies(cpp-targets ${target_name})
endfunction()

# Include all tools
# Auto-discover and include all technology subdirectories (cuda, hip, opencl, etc.)
foreach(tech_dir cuda hip)
    if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/${tech_dir}/benchmarks)
        # Look for tool subdirectories within each technology directory
        file(GLOB tool_dirs LIST_DIRECTORIES true RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/src/${tech_dir}/benchmarks ${CMAKE_CURRENT_SOURCE_DIR}/src/${tech_dir}/benchmarks/*)
        foreach(tool_dir ${tool_dirs})
            if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/${tech_dir}/benchmarks/${tool_dir})
                if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/${tech_dir}/benchmarks/${tool_dir}/CMakeLists.txt)
                    message(STATUS "Including tool: ${tech_dir}/benchmarks/${tool_dir}")
                    add_subdirectory(src/${tech_dir}/benchmarks/${tool_dir})
                endif()
            endif()
        endforeach()
    endif()
endforeach()


# ######### CLANGD COMPILE COMMAND TRICK ********************************
# # The followiing CMake targets are only needed to force CMake to output
# # compile commands in `compile_commands.json` for the fake source files
# # that clangd will use to determine the compile commands for the header
# # files. We call this mechanism the "clangd compile command trick".


# # Find all .cu files in include/cuda directory recursively
# file(GLOB_RECURSE CUDA_SOURCE_FILES
#     "${CMAKE_CURRENT_SOURCE_DIR}/include/cuda/*.cu"
# )

# # Create CUDA library from all .cu files
# if(CUDA_SOURCE_FILES)
#     add_library(cuda_headers_clangd_compile_command_trick STATIC ${CUDA_SOURCE_FILES})
#     set_target_properties(cuda_headers_clangd_compile_command_trick PROPERTIES LINKER_LANGUAGE CUDA)

#     # Set CUDA specific properties for the library
#     set_target_properties(cuda_headers_clangd_compile_command_trick PROPERTIES
#         CUDA_SEPARABLE_COMPILATION ON
#         CUDA_RUNTIME_LIBRARY Shared
#         CUDA_ARCHITECTURES 89
#         LINKER_LANGUAGE CUDA
#     )

#     # Apply CUDA compilation flags
#     separate_arguments(LOCAL_CUDA_FLAGS UNIX_COMMAND ${DEFAULT_CUDA_FLAGS})
#     target_compile_options(cuda_headers_clangd_compile_command_trick PRIVATE ${LOCAL_CUDA_FLAGS})

#         # Set include directories for the library
#     target_include_directories(cuda_headers_clangd_compile_command_trick PUBLIC
#         ${CMAKE_SOURCE_DIR}/include
#         ${CMAKE_SOURCE_DIR}/include/cuda
#         ${CMAKE_SOURCE_DIR}/third-party/cxxopts/include
#         ${CMAKE_SOURCE_DIR}/third-party/cutlass/include
#         ${EIGEN3_INCLUDE_DIR}
#     )

#     # Link CUDA libraries
#     target_link_directories(cuda_headers_clangd_compile_command_trick PUBLIC
#         ${CUDAToolkit_LIBRARY_DIR}
#         ${CUDA_HOST_COMPILER_LIBRARY_PATH}
#     )

#     target_link_libraries(cuda_headers_clangd_compile_command_trick PUBLIC
#         CUDA::cudart
#         CUDA::cuda_driver
#         CUDA::cublas
#         CUDA::cublasLt
#         Eigen3::Eigen
#         OpenMP::OpenMP_CXX
#     )

#         # Add this library to the cuda-targets meta-target
#     add_dependencies(cuda-targets cuda_headers_clangd_compile_command_trick)

#     list(LENGTH CUDA_SOURCE_FILES CUDA_SOURCE_COUNT)
#     message(STATUS "Created CUDA library 'cuda_headers_clangd_compile_command_trick' with ${CUDA_SOURCE_COUNT} source files")
#     foreach(cu_file ${CUDA_SOURCE_FILES})
#         message(STATUS "  - ${cu_file}")
#     endforeach()
# else()
#     message(STATUS "No .cu files found in include/cuda directory")
# endif()

# # Find all .cpp files in include/common directory recursively
# file(GLOB_RECURSE COMMON_SOURCE_FILES
#     "${CMAKE_CURRENT_SOURCE_DIR}/include/common/*.cpp"
# )

# # Create C++ library from all .cpp files
# if(COMMON_SOURCE_FILES)
#     add_library(common_headers_clangd_compile_command_trick STATIC ${COMMON_SOURCE_FILES})
#     set_target_properties(common_headers_clangd_compile_command_trick PROPERTIES LINKER_LANGUAGE CXX)

#     # Set C++ specific properties for the library
#     set_target_properties(common_headers_clangd_compile_command_trick PROPERTIES
#         CXX_STANDARD 20
#         CXX_STANDARD_REQUIRED ON
#     )

#     # Set include directories for the library
#     target_include_directories(common_headers_clangd_compile_command_trick PUBLIC
#         ${CMAKE_SOURCE_DIR}/include
#         ${CMAKE_SOURCE_DIR}/include/common
#         ${CMAKE_SOURCE_DIR}/third-party/cxxopts/include
#         ${EIGEN3_INCLUDE_DIR}
#     )

#     # Link libraries
#     target_link_libraries(common_headers_clangd_compile_command_trick PUBLIC
#         Eigen3::Eigen
#         OpenMP::OpenMP_CXX
#     )

#     # Add this library to the cpp-targets meta-target
#     add_dependencies(cpp-targets common_headers_clangd_compile_command_trick)

#     list(LENGTH COMMON_SOURCE_FILES COMMON_SOURCE_COUNT)
#     message(STATUS "Created C++ library 'common_headers_clangd_compile_command_trick' with ${COMMON_SOURCE_COUNT} source files")
#     foreach(cpp_file ${COMMON_SOURCE_FILES})
#         message(STATUS "  - ${cpp_file}")
#     endforeach()
# else()
#     message(STATUS "No .cpp files found in include/common directory")
# endif()
