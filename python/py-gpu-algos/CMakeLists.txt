# py-gpu-algos Python bindings
# This CMakeLists.txt is a subdirectory of the main project
# All dependencies and settings are inherited from the toplevel CMakeLists.txt

# All dependencies (Python3, pybind11, CUDAToolkit, Eigen3) found by toplevel CMakeLists.txt

# All CUDA flags and include directories inherited from toplevel project

# Matrix product operations CUDA module
pybind11_add_module(_matrix_product_ops_cuda
    src/cuda/matrix_product_ops.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/cuda/cuda_utils.cu
)

# Set module properties
target_compile_definitions(_matrix_product_ops_cuda PRIVATE
    VERSION_INFO=${CMAKE_PROJECT_VERSION}
    WITH_CUDA
)

# Include directories are inherited from toplevel project via include_directories()

# CUDA-specific settings (inherit CUDA_ARCHITECTURES from toplevel project)
set_target_properties(_matrix_product_ops_cuda PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_SEPARABLE_COMPILATION ON
)

# Set CUDA source properties (using flags from toplevel project)
# CUTLASS causes errors due to initializer order mismatch - apply same workaround as main project
set(DEFAULT_CUDA_FLAGS_ORIGINAL "${DEFAULT_CUDA_FLAGS}")
string(REPLACE ",reorder" "" DEFAULT_CUDA_FLAGS "${DEFAULT_CUDA_FLAGS}")
separate_arguments(LOCAL_CUDA_FLAGS UNIX_COMMAND ${DEFAULT_CUDA_FLAGS})
separate_arguments(LOCAL_LINK_FLAGS UNIX_COMMAND ${DEFAULT_LINK_FLAGS})
set_source_files_properties(
    src/cuda/matrix_product_ops.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/cuda/cuda_utils.cu
    PROPERTIES
    LANGUAGE CUDA
    COMPILE_OPTIONS "${LOCAL_CUDA_FLAGS}"
)
# Restore original flags for other targets
set(DEFAULT_CUDA_FLAGS "${DEFAULT_CUDA_FLAGS_ORIGINAL}")

# Apply link options from toplevel project
target_link_options(_matrix_product_ops_cuda PRIVATE "${LOCAL_LINK_FLAGS}")

# Link libraries (matching toplevel project)
target_link_libraries(_matrix_product_ops_cuda PRIVATE
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::cublas
    CUDA::cublasLt
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
)

# Compiler-specific options for C++ only (CUDA flags already set via set_source_files_properties)
target_compile_options(_matrix_product_ops_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
)

# Python module properties (pybind11 handles suffix automatically)
set_target_properties(_matrix_product_ops_cuda PROPERTIES
    OUTPUT_NAME "_matrix_product_ops_cuda"
    PREFIX ""
)

# Matrix transpose operations CUDA module
pybind11_add_module(_matrix_transpose_ops_cuda
    src/cuda/matrix_transpose_ops.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/cuda/cuda_utils.cu
)

# Set module properties
target_compile_definitions(_matrix_transpose_ops_cuda PRIVATE
    VERSION_INFO=${CMAKE_PROJECT_VERSION}
    WITH_CUDA
)

# CUDA-specific settings (inherit CUDA_ARCHITECTURES from toplevel project)
set_target_properties(_matrix_transpose_ops_cuda PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_SEPARABLE_COMPILATION ON
)

# Set CUDA source properties (using flags from toplevel project)
set_source_files_properties(
    src/cuda/matrix_transpose_ops.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/cuda/cuda_utils.cu
    PROPERTIES
    LANGUAGE CUDA
    COMPILE_OPTIONS "${LOCAL_CUDA_FLAGS}"
)

# Apply link options from toplevel project
target_link_options(_matrix_transpose_ops_cuda PRIVATE "${LOCAL_LINK_FLAGS}")

# Link libraries (matching toplevel project)
target_link_libraries(_matrix_transpose_ops_cuda PRIVATE
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::cublas
    CUDA::cublasLt
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
)

# Compiler-specific options for C++ only (CUDA flags already set via set_source_files_properties)
target_compile_options(_matrix_transpose_ops_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
)

# Python module properties (pybind11 handles suffix automatically)
set_target_properties(_matrix_transpose_ops_cuda PROPERTIES
    OUTPUT_NAME "_matrix_transpose_ops_cuda"
    PREFIX ""
)

# Vector operations CUDA module
pybind11_add_module(_vector_ops_cuda
    src/cuda/vector_ops.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/cuda/cuda_utils.cu
)

# Set module properties
target_compile_definitions(_vector_ops_cuda PRIVATE
    VERSION_INFO=${CMAKE_PROJECT_VERSION}
    WITH_CUDA
)

# CUDA-specific settings (inherit CUDA_ARCHITECTURES from toplevel project)
set_target_properties(_vector_ops_cuda PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_SEPARABLE_COMPILATION ON
)

# Set CUDA source properties (using flags from toplevel project)
set_source_files_properties(
    src/cuda/vector_ops.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/cuda/cuda_utils.cu
    PROPERTIES
    LANGUAGE CUDA
    COMPILE_OPTIONS "${LOCAL_CUDA_FLAGS}"
)

# Apply link options from toplevel project
target_link_options(_vector_ops_cuda PRIVATE "${LOCAL_LINK_FLAGS}")

# Link libraries (matching toplevel project)
target_link_libraries(_vector_ops_cuda PRIVATE
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::cublas
    CUDA::cublasLt
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
)

# Compiler-specific options for C++ only (CUDA flags already set via set_source_files_properties)
target_compile_options(_vector_ops_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
)

# Python module properties (pybind11 handles suffix automatically)
set_target_properties(_vector_ops_cuda PROPERTIES
    OUTPUT_NAME "_vector_ops_cuda"
    PREFIX ""
)

# GLM operations CUDA module
pybind11_add_module(_glm_ops_cuda
    src/cuda/glm_ops.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/cuda/cuda_utils.cu
)

# Set module properties
target_compile_definitions(_glm_ops_cuda PRIVATE
    VERSION_INFO=${CMAKE_PROJECT_VERSION}
    WITH_CUDA
)

# CUDA-specific settings (inherit CUDA_ARCHITECTURES from toplevel project)
set_target_properties(_glm_ops_cuda PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_SEPARABLE_COMPILATION ON
)

# Set CUDA source properties (using flags from toplevel project)
set_source_files_properties(
    src/cuda/glm_ops.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/cuda/cuda_utils.cu
    PROPERTIES
    LANGUAGE CUDA
    COMPILE_OPTIONS "${LOCAL_CUDA_FLAGS}"
)

# Apply link options from toplevel project
target_link_options(_glm_ops_cuda PRIVATE "${LOCAL_LINK_FLAGS}")

# Link libraries (matching toplevel project)
target_link_libraries(_glm_ops_cuda PRIVATE
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::cublas
    CUDA::cublasLt
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
)

# Compiler-specific options for C++ only (CUDA flags already set via set_source_files_properties)
target_compile_options(_glm_ops_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
)

# Python module properties (pybind11 handles suffix automatically)
set_target_properties(_glm_ops_cuda PROPERTIES
    OUTPUT_NAME "_glm_ops_cuda"
    PREFIX ""
)

# Sort operations CUDA module
pybind11_add_module(_sort_ops_cuda
    src/cuda/sort_ops.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/cuda/cuda_utils.cu
)

# Set module properties
target_compile_definitions(_sort_ops_cuda PRIVATE
    VERSION_INFO=${CMAKE_PROJECT_VERSION}
    WITH_CUDA
)

# CUDA-specific settings (inherit CUDA_ARCHITECTURES from toplevel project)
set_target_properties(_sort_ops_cuda PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_SEPARABLE_COMPILATION ON
)

# Set CUDA source properties (using flags from toplevel project)
set_source_files_properties(
    src/cuda/sort_ops.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/cuda/cuda_utils.cu
    PROPERTIES
    LANGUAGE CUDA
    COMPILE_OPTIONS "${LOCAL_CUDA_FLAGS}"
)

# Apply link options from toplevel project
target_link_options(_sort_ops_cuda PRIVATE "${LOCAL_LINK_FLAGS}")

# Link libraries (matching toplevel project)
target_link_libraries(_sort_ops_cuda PRIVATE
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::cublas
    CUDA::cublasLt
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
)

# Compiler-specific options for C++ only (CUDA flags already set via set_source_files_properties)
target_compile_options(_sort_ops_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
)

# Python module properties (pybind11 handles suffix automatically)
set_target_properties(_sort_ops_cuda PROPERTIES
    OUTPUT_NAME "_sort_ops_cuda"
    PREFIX ""
)

# Optional: Add to a meta-target for easy building
add_custom_target(py-gpu-algos-modules
    DEPENDS _matrix_product_ops_cuda _matrix_transpose_ops_cuda _vector_ops_cuda _glm_ops_cuda _sort_ops_cuda
    COMMENT "Building py-gpu-algos Python modules"
)

message(STATUS "Configured py-gpu-algos Python bindings")
message(STATUS "  - Using CUDA flags: ${DEFAULT_CUDA_FLAGS}")
