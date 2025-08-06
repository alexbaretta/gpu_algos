<!-- Copyright (c) 2025 Alessandro Baretta -->
<!-- All rights reserved. -->

# GPU Algos

This project contains various numerical algorithms, implemented in both CUDA (for NVidia GPUs), HIP (for AMD GPUs), and plain C++.
Each algorithm is provided as a C++ template and can be referenced in any C++ project by adding `<project_root>/include` to the
include directories. This usually entails adding `-I<project_root>/include` to the compile command of your project.

For each algorithm we provide a C++ driver program that will run the GPU code on randomized inputs of a specified size, and will check
the output of the GPU algorithm vs a plain-vanilla C++ implementation. The driver program will also time both the GPU and C++
versions of the algorithm and compute the speedup.

## Project status

CUDA is well supported, with 19 algorithms provided as C++ header files. For some of these algorithms, multiple
implementations are provided, than can be selected with the `gpu_algo` parameter to the kernel class.

Not all kernels that are available for CUDA are also available for HIP. Nonetheless,
I plan to gradually bring the HIP codebase up to par with CUDA, to the extent possible, acknowledging that some
algorithms might not be portable to HIP. For instance, CUTLASS does not officially support HIP.

Python bindings are not currently available in the master branch, but I am working on them in a development branch.
Now that the C++ APIs have stabilized, I should be able to get the Python part of the codebase to a level where
it could be merged.

Contributions to the GPU Algos project are more than welcome.



## Project Structure
```
.
├── CMakeLists.txt          # Main CMake configuration
├── CMakePresets.json       # CMake presets for build configurations
├── SETUP.md                # Setup and installation instructions
├── .gitmodules             # Git submodules configuration
├── COPYRIGHT.txt           # Copyright information
├── include/                # Header files
│   ├── common/             # C++ headers
│   ├── cuda/               # CUDA-specific headers
│   └── hip/                # HIP-specific headers
├── src/                    # Source files
│   ├── common/             # C++ source files
│   ├── cuda/               # CUDA implementations
│   └── hip/                # HIP implementations
├── builds/                 # Build output directory
│   ├── release/            # Release builds
│   └── debug/              # Debug builds
├── test/                   # GPU Algorithm Test Framework - comprehensive testing toolkit
│   ├── gpu_algo_test/      # Main Python package with testing framework
│   ├── tests/              # Python unit tests
│   ├── pyproject.toml      # Python package configuration
│   └── README.md           # Python-specific documentation
├── scripts/                # Build and utility scripts
├── docs/                   # Documentation
└── third-party/            # External dependencies
```

## Requirements
- CUDA Toolkit 12.9
- CMake 3.18 or higher
- C++17 compatible compiler
- NVIDIA GPU for the CUDA algorithms
- AMD GPU for the HIP algorithms

## Building the Project
The build process is configured via cmake. Bash scripts are provided to simplify the configuration and build process, although you can build the project using cmake directly.

1. Configure the build directory, both debug and release builds.

The following script configures the both the debug and release builds.
```bash
./scripts/configure_build.sh
```

2. Build the project:
The following scripts run the release and debug builds respectively.

```bash
./scripts/release_build.sh # Release build
./scripts/debug_build.sh   # Debug build
```

Any argument is passed on to cmake, allowing you to use cmake CLI options with their usual meanings. For example, the following command shows how to build a single executable, using a single compiler proecess, with verbose output.
```bash
./scripts/debug_build.sh -j1 --target matrix_product_naive --verbose
```


## Running the Algorithms

After building, the executables will be located in `builds/release/bin/` (or `builds/debug/bin/` for debug builds). Each algorithm has one or more implementations with different optimization strategies.

### Available Algorithms

#### Matrix Operations
- **Matrix Product (Multiplication)**:
  ```bash
  matrix_product_naive      # One thread per output value
  matrix_product_warp       # One warp per output value
  matrix_product_tiled      # Tiled/shared memory optimization
  matrix_product_tensor     # WMMA Tensor Core implementation
  matrix_product_cublas     # cuBLASlt reference
  matrix_product_cutlass    # CUTLASS implementation
  ```

- **Matrix Transpose**:
  ```bash
  matrix_transpose_naive    # One thread per output value
  matrix_transpose_striped  # Memory coalescing optimization
  matrix_transpose_tiled    # Tiled memory access
  matrix_transpose_cublas   # cuBLAS reference
  ```

#### Vector Operations
- **Vector Cumulative Sum (Prefix Sum)**:
  ```bash
  vector_cumsum_serial      # Serial algorithm, slow
  vector_cumsum_parallel    # Parallel prefix sum
  ```

- **Vector Cumulative Maximum**:
  ```bash
  vector_cummax_parallel    # Parallel cumulative maximum
  ```

- **Vector Scan (Generic)**:
  ```bash
  vector_scan_parallel      # Generic parallel scan operation
  ```

- **Tensor Sort**:
  ```bash
  tensor_sort_bitonic       # Sort tensors along one dimension
  ```

- **Generalized Linear Models (GLM)**:
Currently, only linear link with L2 loss and L2 penalty are supported.
  ```bash
  glm_predict_naive         # Compute a GLM prediction
  glm_gradient_naive        # Basic gradient computation
  glm_gradient_xyyhat       # Optimized gradient computation
  ```

### Benchmarks
Although the project is intended to be used as C++ header library, the build system produces a number of executables, which are intended for testing and performance benchmarking. Use `-h` on each executable to get a listing of CLI options. Generally, the following CLI options are supported
1. -h, --help: Print usage instructions and list CLI options
2. --type: Select numeric data type (half, float, double, int<n>, uint<n>, where n is 8, 16, 32, 64)
3. -M, -N, -K, -T...: Single capital letter options are used to configure the problem size (length of a vector, matrix size, tensor dimentsion, ...)
4. --init-method increasing: Instead of randomizing the input data, generate an increasing sequence
4. --block-dim-x, --block-dim-y, --block-dim-z: The size of the GPU thread block in each dimension.
5. --block-dim: The size of 1-dimensional GPU thread block.
6. --tol-bits: how many bits of precison are lost due to floating point rounding.


### Example Usage

```bash
# Run matrix multiplication with custom size
./builds/release/bin/matrix_product_tiled -M 2000 -K 1000 -N 1500

# Run vector cumsum with verbose output
./builds/release/bin/vector_cumsum_parallel -N 10 --verbose

# Compare different matrix transpose implementations
./builds/release/bin/matrix_transpose_naive -M 25 -N 8 --init-method increasing --type uint64
```

### Program Output

Each program will:
1. Initialize input data with randomized or sequential values of the specified size
2. Run the GPU algorithm and measure execution time
3. Run a reference C++ implementation for correctness verification
4. Compare results and report any discrepancies
5. Display timing results and compute the GPU speedup over CPU

Sample output:
```bash
$ ./builds/release/bin/matrix_transpose_naive -M 2500 -N 8150 --type half
Input matrix A dimensions   : 2500x8150
Output matrix dimensions    : 8150x2500
Temp matrix dimensions      : 0x0
Input size                  : 0.0379514 GB (40750000 bytes)
Output size                 : 0.0379514 GB (40750000 bytes)
Temp size                   : 0 GB (0 bytes)
Required memory             : 0.0759028 GB (81500000 bytes)

SETUP:
  - Allocating memory: 19.1607 ms (19.1607 ms total)
  - Initializing matrices:   (random) 148.768 ms (167.929 ms total)
  - Creating GPU streams: 71.4346 ms (239.363 ms total)
  - Creating GPU events: 0.015961 ms (239.379 ms total)
Matrix_Kernel_1In_1Out:
1 -  cudaEventElapsedTime    Allocate device memory: 3.13998 ms (3.13998 ms total)
1 - std::chrono::duration    Allocate device memory: 2.83814 ms (2.83814 ms total)
2 -  cudaEventElapsedTime       Copy data to device: 3.65555 ms (6.79553 ms total)
2 - std::chrono::duration       Copy data to device: 3.77645 ms (6.61459 ms total)
3 -  cudaEventElapsedTime            Compute kernel: 0.444054 ms (7.23959 ms total)
3 - std::chrono::duration            Compute kernel: 0.49056 ms (7.10515 ms total)
4 -  cudaEventElapsedTime  Copy result back to host: 3.31543 ms (10.555 ms total)
4 - std::chrono::duration  Copy result back to host: 3.31488 ms (10.42 ms total)
5 -  cudaEventElapsedTime        Free device memory: 1.2967 ms (11.8517 ms total)
5 - std::chrono::duration        Free device memory: 0.131072 ms (10.5511 ms total)
CHECK WITH CPU:
 -      Convert data to Eigen: 0.00046 ms (0.00046 ms total)
 -  Compute result with Eigen: 22.7187 ms (22.7192 ms total)
 -       Compute error matrix: 115.627 ms (138.346 ms total)
 -          Compute max error: 81.6959 ms (220.042 ms total)
DONE: 471.313 ms total
Max error     : 0 at (0, 0)
Max error pct : 0 at (0, 0)
Precision     : 0.000488281 (CUDA __half with 11 bits of precision)
Tolerance pct : 0.0078125% assuming a loss of 4 bits of precision
Gross speedup : 46.3118
Net speedup   : 2.15325
[SUCCESS]     : Max error pct is within tolerance
```

## Comprehensive Testing with gpu_algo_test

For extensive testing and validation of all GPU algorithms, this project includes a sophisticated Python testing framework called `gpu_algo_test`. This framework can automatically discover and test all built executables with various problem sizes and data types, making it ideal for regression testing and performance evaluation.

For example, to run a comprehensive set of benchmarks, using only CUDA executables from the release build, recording test results as a stream of JSON objects, one per line, in `output.jsonl`, and with detailed output to the terminal:
```bash
python test/gpu_algo_test/gpu_algo_test.py --cuda-only --preset release --output output.jsonl
```

### Quick Start - Test Everything

To run a comprehensive test of all algorithms with default settings:


### Testing different GPU problem sizes and data types
The framework provides predefined problem size groups that test different categories problem sizes. The objective is to ensure that the algorithms work correctly both when the problem size is very small and when it is as large as is realistic given the available memory and compute power, and when the problem size is a multiple of a warp or of a typical block size, and when it includes partial blocks at the borders of the grid.

Here are a few examples.
```bash
# Test algorithms with sub-optimal GPU utilization (< 32 elements)
python test/gpu_algo_test/gpu_algo_test.py --sizes smaller_than_warp --types floating

# Test optimal single-warp utilization (32 elements)
python test/gpu_algo_test/gpu_algo_test.py --sizes exactly_one_warp --types floating,integer

# Test small multi-warp grid
python test/gpu_algo_test/gpu_algo_test.py --sizes several_warps_boundary,several_warps_partial

# Test full thread block utilization (1024 elements)
python test/gpu_algo_test/gpu_algo_test.py --sizes full_block --types floating

# Test multi-block grid with maximum problem size of 1e9 (the product of all problem dimemensions)
python test/gpu_algo_test/gpu_algo_test.py --sizes several_blocks_boundary,several_blocks_partial --max-problem-size 1000000000
```

### Selective Algorithm Testing

Test specific algorithms or algorithm families:

```bash
# Test a single executable
python test/gpu_algo_test/gpu_algo_test.py --executables matrix_product_tensor

# Test only executables matching globs
python test/gpu_algo_test/gpu_algo_test.py --executables matrix_*,tensor_*

# Exclude executables matching globs
python test/gpu_algo_test/gpu_algo_test.py --executables -matrix_*,-tensor*

```

### Data Type Coverage

Test algorithms across different numerical precisions:

```bash
# Test floating-point precision variations
python test/gpu_algo_test/gpu_algo_test.py --types floating  # half, float, double

# Test integer type variations
python test/gpu_algo_test/gpu_algo_test.py --types integer   # int8, int16, int32, int64, uint8, uint16, uint32, uint64

# Test specific data types
python test/gpu_algo_test/gpu_algo_test.py --types float,double --sizes 1024,2048
```

### Rerunning previously failed tests

You can rerun only tests that are recorded as failed in the JSONL output file of a previous run.
```
python test/gpu_algo_test/gpu_algo_test.py --cuda-only --preset release --verbose \
      --rerun-failures release.jsonl --output release.rerun.jsonl
```

### All options

```
  -h, --help            show this help message and exit
  --bin-path BIN_PATH   Direct path to bin directory (overrides CMake preset logic) (default: None)
  --cmake-root CMAKE_ROOT
                        Path to CMake root directory (defaults to current directory) (default: None)
  --preset PRESET       CMake preset name (defaults to 'debug') (default: debug)
  -v, --verbose         Enable verbose output (default: False)
  -o OUTPUT, --output OUTPUT
                        Output file for detailed results (JSON format) (default: None)
  --executables EXECUTABLES
                        Comma-separated list of executable names or glob patterns to test (default: all). Supports shell-style wildcards: * matches any number of characters, ? matches a single character. Examples: 'vector_*' matches all executables
                        starting with 'vector_', '*_sort' matches all executables ending with '_sort' (default: None)
  --sizes SIZES         Comma-separated list of problem sizes to test (default: all). Can include numbers or special size categories: smaller_than_warp, exactly_one_warp, several_warps_boundary, several_warps_partial, full_block,
                        several_blocks_boundary, several_blocks_partial (default: None)
  --max-problem-size MAX_PROBLEM_SIZE
                        Maximum total size of the problem (product of all dimensions). (default: 268435456)
  --types TYPES         Comma-separated list of data types to test (default: all). Can include specific types or special groups: floating, signed, unsigned, integer. Available types: half, float, double, int8, int16, int32, int64, uint8, uint16,
                        uint32, uint64 (default: None)
  --dryrun              Only check for executable existence, don't run tests (default: False)
  --tol-bits TOL_BITS   Number of bits of precision loss for floating point tolerance (default: 6) (default: 6)
  --rerun-failures RERUN_FAILURES
                        Path to previous test results file (JSONL format) to rerun only failed tests (default: None)
  --timeout TIMEOUT     Timeout for test execution in seconds (default: 300 seconds) (default: 30)
  --hip-only            Only test HIP executables (executables with 'hip_' prefix) (default: False)
  --cuda-only           Only test CUDA executables (executables without 'hip_' prefix) (default: False)
  ```

### Expected Test Output
Here is an example of the output produced by gpu_algo_test on the console.
```
================================================================================
GPU Algorithm Test Report
================================================================================

glm_gradient_xyyhat:
--------------------
  Tests: 48 total, 48 executed successfully, 0 execution failures
  Correctness: 48 correct

matrix_product_cublas:
----------------------
  Tests: 1 total, 1 executed successfully, 0 execution failures
  Correctness: 0 correct, 1 correctness failures

matrix_product_cutlass:
-----------------------
  Tests: 6 total, 6 executed successfully, 0 execution failures
  Correctness: 0 correct, 6 correctness failures

matrix_product_naive:
---------------------
  Tests: 6 total, 6 executed successfully, 0 execution failures
  Correctness: 0 correct, 6 correctness failures

matrix_product_tensor:
----------------------
  Tests: 1 total, 1 executed successfully, 0 execution failures
  Correctness: 0 correct, 1 correctness failures

matrix_product_tiled:
---------------------
  Tests: 6 total, 6 executed successfully, 0 execution failures
  Correctness: 0 correct, 6 correctness failures

================================================================================
Overall: 48/68 tests passed both execution and correctness checks (70.6%)
================================================================================

================================================================================
TESTS THAT FAILED CORRECTNESS CHECKS
================================================================================

Total correctness failures: 20

Command lines:
  1. matrix_product_cublas (type=double, size_i=0, sizes=[2048, 3072])
     Command: /home/alex/git/gpu_algos/builds/release/bin/matrix_product_cublas -M 2048 -K 3072 -N 42 --type double --tol-bits 6
     Max error: 5.57066e-12, Max error %: 7.2224e-13
  2. matrix_product_cutlass (type=double, size_i=0, sizes=[2048, 3072])
     Command: /home/alex/git/gpu_algos/builds/release/bin/matrix_product_cutlass -M 2048 -K 3072 -N 42 --type double --tol-bits 6
     Max error: 5.57066e-12, Max error %: 7.2224e-13
  3. matrix_product_cutlass (type=half, size_i=0, sizes=[2048, 3072])
     Command: /home/alex/git/gpu_algos/builds/release/bin/matrix_product_cutlass -M 2048 -K 3072 -N 42 --type half --tol-bits 6
     Max error: 64.0, Max error %: 8.16327
  4. matrix_product_cutlass (type=float, size_i=0, sizes=[2048, 3072])
     Command: /home/alex/git/gpu_algos/builds/release/bin/matrix_product_cutlass -M 2048 -K 3072 -N 42 --type float --tol-bits 6
     Max error: 0.00323486, Max error %: 0.000412834
  5. matrix_product_cutlass (type=half, size_i=1, sizes=[2048, 3072])
     Command: /home/alex/git/gpu_algos/builds/release/bin/matrix_product_cutlass -M 3072 -K 2048 -N 42 --type half --tol-bits 6
     Max error: 22.5, Max error %: 4.38169
  6. matrix_product_cutlass (type=half, size_i=0, sizes=[1025, 1500, 2500])
     Command: /home/alex/git/gpu_algos/builds/release/bin/matrix_product_cutlass -M 1025 -K 1500 -N 174 --type half --tol-bits 6
     Max error: 14.0, Max error %: 3.66252
  7. matrix_product_cutlass (type=half, size_i=1, sizes=[1025, 1500, 2500])
     Command: /home/alex/git/gpu_algos/builds/release/bin/matrix_product_cutlass -M 1500 -K 2500 -N 71 --type half --tol-bits 6
     Max error: 40.5, Max error %: 6.53226
  8. matrix_product_naive (type=double, size_i=0, sizes=[2048, 3072])
     Command: /home/alex/git/gpu_algos/builds/release/bin/matrix_product_naive -M 2048 -K 3072 -N 42 --type double --tol-bits 6
     Max error: 5.57066e-12, Max error %: 7.2224e-13
  9. matrix_product_naive (type=half, size_i=0, sizes=[2048, 3072])
     Command: /home/alex/git/gpu_algos/builds/release/bin/matrix_product_naive -M 2048 -K 3072 -N 42 --type half --tol-bits 6
     Max error: 64.0, Max error %: 8.16327
 10. matrix_product_naive (type=float, size_i=0, sizes=[2048, 3072])
     Command: /home/alex/git/gpu_algos/builds/release/bin/matrix_product_naive -M 2048 -K 3072 -N 42 --type float --tol-bits 6
     Max error: 0.00323486, Max error %: 0.000412834
 11. matrix_product_naive (type=half, size_i=1, sizes=[2048, 3072])
     Command: /home/alex/git/gpu_algos/builds/release/bin/matrix_product_naive -M 3072 -K 2048 -N 42 --type half --tol-bits 6
     Max error: 22.5, Max error %: 4.38169
 12. matrix_product_naive (type=half, size_i=0, sizes=[1025, 1500, 2500])
     Command: /home/alex/git/gpu_algos/builds/release/bin/matrix_product_naive -M 1025 -K 1500 -N 174 --type half --tol-bits 6
     Max error: 14.0, Max error %: 3.66252
 13. matrix_product_naive (type=half, size_i=1, sizes=[1025, 1500, 2500])
     Command: /home/alex/git/gpu_algos/builds/release/bin/matrix_product_naive -M 1500 -K 2500 -N 71 --type half --tol-bits 6
     Max error: 40.5, Max error %: 6.53226
 14. matrix_product_tensor (type=double, size_i=0, sizes=[2048, 3072])
     Command: /home/alex/git/gpu_algos/builds/release/bin/matrix_product_tensor -M 2048 -K 3072 -N 42 --type double --tol-bits 6
     Max error: 5.57066e-12, Max error %: 7.2224e-13
 15. matrix_product_tiled (type=double, size_i=0, sizes=[2048, 3072])
     Command: /home/alex/git/gpu_algos/builds/release/bin/matrix_product_tiled -M 2048 -K 3072 -N 42 --type double --tol-bits 6
     Max error: 5.57066e-12, Max error %: 7.2224e-13
 16. matrix_product_tiled (type=half, size_i=0, sizes=[2048, 3072])
     Command: /home/alex/git/gpu_algos/builds/release/bin/matrix_product_tiled -M 2048 -K 3072 -N 42 --type half --tol-bits 6
     Max error: 64.0, Max error %: 8.16327
 17. matrix_product_tiled (type=float, size_i=0, sizes=[2048, 3072])
     Command: /home/alex/git/gpu_algos/builds/release/bin/matrix_product_tiled -M 2048 -K 3072 -N 42 --type float --tol-bits 6
     Max error: 0.00323486, Max error %: 0.000412834
 18. matrix_product_tiled (type=half, size_i=1, sizes=[2048, 3072])
     Command: /home/alex/git/gpu_algos/builds/release/bin/matrix_product_tiled -M 3072 -K 2048 -N 42 --type half --tol-bits 6
     Max error: 22.5, Max error %: 4.38169
 19. matrix_product_tiled (type=half, size_i=0, sizes=[1025, 1500, 2500])
     Command: /home/alex/git/gpu_algos/builds/release/bin/matrix_product_tiled -M 1025 -K 1500 -N 174 --type half --tol-bits 6
     Max error: 14.0, Max error %: 3.66252
 20. matrix_product_tiled (type=half, size_i=1, sizes=[1025, 1500, 2500])
     Command: /home/alex/git/gpu_algos/builds/release/bin/matrix_product_tiled -M 1500 -K 2500 -N 71 --type half --tol-bits 6
     Max error: 40.5, Max error %: 6.53226
================================================================================
BENCHMARK SUMMARY
================================================================================
Total benchmarks: 68
Execution failures: 0
Successful executions: 68
Correctness failures: 20
Correct results: 48
```

This testing framework ensures comprehensive validation of all GPU algorithms across different problem sizes, data types, and GPU utilization patterns, making it an essential tool for maintaining code quality and performance optimization.

# Bugs and workarounds

## Eigen infinite loop when OpenMP enabled in HIP executables

Some Eigen operations trigger an infinite loop when using OpenMP in HIP executables. Here is the deadlocked thread's stack trace, reported by gdb.
```
#0  std::__atomic_base<int>::load (this=0x7ffd25290248, __m=std::memory_order::seq_cst)
    at /usr/lib/gcc/x86_64-linux-gnu/13/../../../../include/c++/13/bits/atomic_base.h:505
#1  std::__atomic_base<int>::operator int (this=0x7ffd25290248)
    at /usr/lib/gcc/x86_64-linux-gnu/13/../../../../include/c++/13/bits/atomic_base.h:365
#2  Eigen::internal::general_matrix_matrix_product<long, float, 1, false, float, 1, false, 0, 1>::run (rows=64, cols=128,
    depth=96, _lhs=0x14679dd0, lhsStride=96, _rhs=0x1467fde0, rhsStride=128, _res=0x14e01b00, resIncr=1, resStride=64,
    alpha=<error reading variable: That operation is not available on integers of more than 8 bytes.>, blocking=...,
    info=0x7ffd25290240) at /usr/include/eigen3/Eigen/src/Core/products/GeneralMatrixMatrix.h:112
#3  0x000000000022b788 in Eigen::internal::gemm_functor<float, long, Eigen::internal::general_matrix_matrix_product<long, float, 1, false, float, 1, false, 0, 1>, Eigen::Map<Eigen::Matrix<float, -1, -1, 1, -1, -1>, 0, Eigen::Stride<0, 0> >, Eigen::Map<Eigen::Matrix<float, -1, -1, 1, -1, -1>, 0, Eigen::Stride<0, 0> >, Eigen::Matrix<float, -1, -1, 0, -1, -1>, Eigen::internal::gemm_blocking_space<0, float, float, -1, -1, -1, 1, false> >::operator() (this=<optimized out>,
    row=<optimized out>, rows=<optimized out>, col=<optimized out>, cols=<optimized out>, info=<optimized out>)
    at /usr/include/eigen3/Eigen/src/Core/products/GeneralMatrixMatrix.h:230
#4  0x000000000022b788 in _ZN5Eigen8internal16parallelize_gemmILb1ENS0_12gemm_functorIflNS0_29general_matrix_matrix_productIlfLi1ELb0EfLi1ELb0ELi0ELi1EEENS_3MapINS_6MatrixIfLin1ELin1ELi1ELin1ELin1EEELi0ENS_6StrideILi0ELi0EEEEESA_NS6_IfLin1ELin1ELi0ELin1ELin1EEENS0_19gemm_blocking_spaceILi0EffLin1ELin1ELin1ELi1ELb0EEEEElEEvRKT0_T1_SI_SI_b.omp_outlined_debug__ (
    cols=<optimized out>, rows=<optimized out>, info=<optimized out>, transpose=<optimized out>, func=...,
    .global_tid.=<optimized out>, .bound_tid.=<optimized out>)
#5  _ZN5Eigen8internal16parallelize_gemmILb1ENS0_12gemm_functorIflNS0_29general_matrix_matrix_productIlfLi1ELb0EfLi1ELb0ELi0ELi1EEENS_3MapINS_6MatrixIfLin1ELin1ELi1ELin1ELin1EEELi0ENS_6StrideILi0ELi0EEEEESA_NS6_IfLin1ELin1ELi0ELin1ELin1EEENS0_19gemm_blocking_spaceILi0EffLin1ELin1ELin1ELi1ELb0EEEEElEEvRKT0_T1_SI_SI_b.omp_outlined(void) const (
    .global_tid.=<optimized out>, .bound_tid.=<optimized out>, cols=<optimized out>, rows=<optimized out>,
    info=<optimized out>, transpose=<optimized out>, func=...)
    at /usr/include/eigen3/Eigen/src/Core/products/Parallelizer.h:151
#6  0x00007f86589eb329 in __kmp_invoke_microtask () from /opt/rocm-6.4.1/lib/llvm/bin/../lib/libomp.so
#7  0x00007f865896a30f in __kmp_invoke_task_func () from /opt/rocm-6.4.1/lib/llvm/bin/../lib/libomp.so
#8  0x00007f8658968f6c in __kmp_launch_thread () from /opt/rocm-6.4.1/lib/llvm/bin/../lib/libomp.so
#9  0x00007f86589cb708 in __kmp_launch_worker(void*) () from /opt/rocm-6.4.1/lib/llvm/bin/../lib/libomp.so
#10 0x00007f86584a81f5 in start_thread (arg=<optimized out>) at ./nptl/pthread_create.c:442
#11 0x00007f865852889c in clone3 () at ../sysdeps/unix/sysv/linux/x86_64/clone3.S:81
```

This seems to be an Eigen bug. For the time being, the workaround is to disable OpenMP for all HIP executables.
