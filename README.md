<!-- Copyright (c) 2025 Alessandro Baretta -->
<!-- All rights reserved. -->

# GPU Algos

This project contains various numerical algorithms, implemented in both CUDA (for NVidia GPUs), HIP (for AMD GPUs), and plain C++.
Each algorithm is provided as a C++ template and can be referenced in any C++ project by adding `<project_root>/include` to the
include directories. This usually entails adding `-I<project_root>/include` to the compile command of your project.

For each algorithm we provide a C++ driver program that will run the GPU code on randomized inputs of a specified size, and will check
the output of the GPU algorithm vs a plain-vanilla C++ implementation. The driver program will also time both the GPU and C++
versions of the algorithm and compute the speedup.

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

1. Configure the build directory, both debug and release builds.
```bash
./scripts/configure_build.sh
```

2. Build the project:
* Release build
```bash
./scripts/release_build.sh
```

* Debug build
```bash
./scripts/debug_build.sh
```


## Running the Algorithms

After building, the executables will be located in `builds/release/bin/` (or `builds/debug/bin/` for debug builds). Each algorithm has one or more implementations with different optimization strategies.

### Available Algorithms

#### Matrix Operations
- **Matrix Product (Multiplication)**:
  ```bash
  ./builds/release/bin/matrix_product_naive      # Basic implementation
  ./builds/release/bin/matrix_product_tiled      # Tiled/shared memory optimization
  ./builds/release/bin/matrix_product_warp       # Warp-level optimization
  ./builds/release/bin/matrix_product_tensor     # Tensor core implementation
  ./builds/release/bin/matrix_product_cublas     # cuBLAS reference
  ./builds/release/bin/matrix_product_cutlass    # CUTLASS implementation
  ```

- **Matrix Transpose**:
  ```bash
  ./builds/release/bin/matrix_transpose_naive    # Basic implementation
  ./builds/release/bin/matrix_transpose_striped  # Memory coalescing optimization
  ./builds/release/bin/matrix_transpose_tiled    # Tiled memory access
  ./builds/release/bin/matrix_transpose_cublas   # cuBLAS reference
  ```

#### Vector Operations
- **Vector Cumulative Sum (Prefix Sum)**:
  ```bash
  ./builds/release/bin/vector_cumsum_serial      # Serial implementation
  ./builds/release/bin/vector_cumsum_parallel    # Parallel prefix sum
  ```

- **Vector Cumulative Maximum**:
  ```bash
  ./builds/release/bin/vector_cummax_parallel    # Parallel cumulative maximum
  ```

- **Vector Scan (Generic)**:
  ```bash
  ./builds/release/bin/vector_scan_parallel      # Generic parallel scan operation
  ```

#### Advanced Operations
- **Tensor Sort**:
  ```bash
  ./builds/release/bin/tensor_sort_bitonic       # Bitonic sort for tensors
  ```

- **Generalized Linear Models (GLM)**:
  ```bash
  ./builds/release/bin/glm_predict_naive         # Basic GLM prediction
  ./builds/release/bin/glm_gradient_naive        # Basic gradient computation
  ./builds/release/bin/glm_gradient_xyyhat       # Optimized gradient computation
  ```

### Command Line Options

Most programs accept the following options:
- `--size N` or `-s N`: Set the problem size (default varies by algorithm)
- `--iterations N` or `-i N`: Number of benchmark iterations (default: 10)
- `--verbose` or `-v`: Enable verbose output
- `--help` or `-h`: Show usage information

### Example Usage

```bash
# Run matrix multiplication with custom size
./builds/release/bin/matrix_product_tiled --size 2048 --iterations 5

# Run vector cumsum with verbose output
./builds/release/bin/vector_cumsum_parallel --size 1000000 --verbose

# Compare different matrix transpose implementations
./builds/release/bin/matrix_transpose_naive --size 4096
./builds/release/bin/matrix_transpose_tiled --size 4096
```

### Program Output

Each program will:
1. Initialize input data with randomized values of the specified size
2. Run the GPU algorithm and measure execution time
3. Run a reference C++ implementation for correctness verification
4. Compare results and report any discrepancies
5. Display timing results and compute the GPU speedup over CPU

Sample output:
```
Matrix size: 2048x2048
GPU time: 2.34 ms
CPU time: 1847.23 ms
Speedup: 789.2x
Result verification: PASSED
```

## Comprehensive Testing with gpu_algo_test

For extensive testing and validation of all GPU algorithms, this project includes a sophisticated Python testing framework called `gpu_algo_test`. This framework can automatically discover and test all built executables with various problem sizes and data types, making it ideal for regression testing, performance validation, and GPU workload analysis.

### Installation

Navigate to the Python directory and install the testing framework:

```bash
cd python
pip install -e .
```

### Quick Start - Test Everything

To run a comprehensive test of all algorithms with default settings:

```bash
# Test all executables with debug build (default)
gpu-algo-test

# Test all executables with release build for performance analysis
gpu-algo-test --preset release
```

### GPU Workload Pattern Testing

The framework provides predefined problem size groups that test different GPU utilization patterns:

```bash
# Test algorithms with sub-optimal GPU utilization (< 32 elements)
gpu-algo-test --sizes smaller_than_warp --types floating

# Test optimal single-warp utilization (32 elements)
gpu-algo-test --sizes exactly_one_warp --types floating,integer

# Test multi-warp coordination within a block
gpu-algo-test --sizes several_warps_boundary,several_warps_partial

# Test full thread block utilization (1024 elements)
gpu-algo-test --sizes full_block --types floating

# Test multi-block grid coordination
gpu-algo-test --sizes several_blocks_boundary,several_blocks_partial
```

### Selective Algorithm Testing

Test specific algorithms or algorithm families:

```bash
# Test only matrix operations
gpu-algo-test --executables matrix_product_naive,matrix_product_tiled,matrix_transpose_naive

# Test vector operations with specific sizes
gpu-algo-test --executables vector_cumsum_parallel,vector_cummax_parallel --sizes 32,64,1024

# Test a single algorithm thoroughly
gpu-algo-test --executables matrix_product_tensor --sizes smaller_than_warp,exactly_one_warp,full_block --types floating
```

### Data Type Coverage

Test algorithms across different numerical precisions:

```bash
# Test floating-point precision variations
gpu-algo-test --types floating  # half, float, double

# Test integer type variations
gpu-algo-test --types integer   # int8, int16, int32, int64, uint8, uint16, uint32, uint64

# Test specific data types
gpu-algo-test --types float,double --sizes 1024,2048
```

### Development and Debugging

Useful options for development and troubleshooting:

```bash
# Dry run to verify executable discovery without running tests
gpu-algo-test --dryrun --verbose

# Save detailed results for later analysis
gpu-algo-test --output test_results.json --preset release

# Test from a custom binary location
gpu-algo-test --bin-path /path/to/custom/bin --verbose

# Verbose output for debugging test failures
gpu-algo-test --executables vector_cumsum_parallel --sizes 32 --types float --verbose
```

### Integration in CI/CD

The framework is designed for automated testing pipelines:

```bash
# Comprehensive regression test (exit code indicates pass/fail)
gpu-algo-test --preset release --output ci_results.json

# Performance regression detection
gpu-algo-test --preset release --types float,double --sizes full_block,several_blocks_boundary
```

### Available Problem Size Groups

- `smaller_than_warp`: [8, 16, 24] - Tests sub-warp algorithms
- `exactly_one_warp`: [32] - Tests optimal warp utilization
- `several_warps_boundary`: [64, 96, 128] - Tests multiple warps at boundaries
- `several_warps_partial`: [33, 50, 100] - Tests partial warp utilization
- `full_block`: [1024] - Tests full thread block algorithms
- `several_blocks_boundary`: [2048, 3072] - Tests multi-block at boundaries
- `several_blocks_partial`: [1025, 1500, 2500] - Tests partial block utilization

### Available Data Type Groups

- `floating`: half, float, double
- `signed`: int8, int16, int32, int64
- `unsigned`: uint8, uint16, uint32, uint64
- `integer`: all integer types (signed + unsigned)

### Expected Test Output

The framework provides detailed reporting:

```
Testing matrix_product_tiled with size=1024, type=float
PASSED - Execution time: 2.34ms, Accuracy: ✓
PASSED - GPU vs CPU verification successful

Testing vector_cumsum_parallel with size=32, type=double
PASSED - Execution time: 0.12ms, Accuracy: ✓
PASSED - Single-warp optimal utilization confirmed

=== Test Summary ===
Total tests: 156
Passed: 154
Failed: 2
Success rate: 98.7%
```

This testing framework ensures comprehensive validation of all GPU algorithms across different problem sizes, data types, and GPU utilization patterns, making it an essential tool for maintaining code quality and performance optimization.
