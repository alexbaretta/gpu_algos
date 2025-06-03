# GPU Algorithm Test Framework

A comprehensive testing framework for GPU algorithm executables with support for various problem sizes and data types.

## Overview

This framework provides functionality to test GPU algorithm executables with various problem sizes designed to test different GPU workload patterns:

- **Sub-warp sizes** (< 32 elements): Test algorithms with minimal parallelism
- **Single warp** (32 elements): Test optimal warp utilization
- **Multiple warps** (64, 96, 128+ elements): Test warp coordination
- **Single block** (1024 elements): Test block-level algorithms
- **Multiple blocks** (2048+ elements): Test grid-level coordination

For each problem size, the framework tests all supported data types including floating-point and integer types.

## Features

- 🚀 **CMake Integration**: Automatic discovery of executables using CMake presets
- 🎯 **Flexible Testing**: Support for custom problem sizes and data types
- 📊 **Performance Metrics**: Automatic extraction of timing and accuracy metrics
- 🔧 **Special Size Groups**: Convenient shortcuts for common GPU workload patterns
- 📋 **Comprehensive Reports**: Detailed test results with failure analysis
- 🐍 **Python API**: Use as both command-line tool and Python package

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/alessandrobaretta/gpu-algos.git
cd gpu-algos/python

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### As a Package

```bash
# Install from the python directory
cd gpu-algos/python
pip install .
```

## Quick Start

### Command Line Usage

```bash
# Test all executables with default settings (debug preset)
gpu-algo-test

# Test with specific executable and problem sizes
gpu-algo-test --executables vector_cumsum_parallel --sizes exactly_one_warp,full_block --types floating

# Use specific CMake preset
gpu-algo-test --preset release --sizes smaller_than_warp,several_warps_boundary

# Use direct binary path (bypasses CMake preset logic)
gpu-algo-test --bin-path /path/to/bin --dryrun
```

### Python API Usage

```python
from gpu_algo_test import GPUAlgoTest

# Create test runner
tester = GPUAlgoTest(
    bin_dir="/path/to/bin",
    selected_executables={"vector_cumsum_parallel"},
    selected_sizes={32, 64, 1024},
    selected_types={"float", "double"}
)

# Run tests
results = tester.run_all_tests()

# Generate report
report = tester.generate_report(results)
print(report)
```

## Command Line Options

### Directory Configuration

- `--bin-path PATH`: Direct path to bin directory (overrides CMake preset logic)
- `--cmake-root PATH`: Path to CMake root directory (defaults to current directory)
- `--preset NAME`: CMake preset name (defaults to 'debug')

### Test Selection

- `--executables LIST`: Comma-separated list of executable names to test
- `--sizes LIST`: Problem sizes to test (numbers or special groups)
- `--types LIST`: Data types to test (specific types or special groups)

### Special Size Groups

- `smaller_than_warp`: [8, 16, 24]
- `exactly_one_warp`: [32]
- `several_warps_boundary`: [64, 96, 128]
- `several_warps_partial`: [33, 50, 100]
- `full_block`: [1024]
- `several_blocks_boundary`: [2048, 3072]
- `several_blocks_partial`: [1025, 1500, 2500]

### Special Type Groups

- `floating`: half, float, double
- `signed`: int8, int16, int32, int64
- `unsigned`: uint8, uint16, uint32, uint64
- `integer`: all integer types (signed + unsigned)

### Output Options

- `-v, --verbose`: Enable verbose output
- `-o OUTPUT, --output OUTPUT`: Save detailed results to JSON file
- `--dryrun`: Only check executable existence, don't run tests

## Examples

### Testing Specific GPU Workload Patterns

```bash
# Test sub-warp and single-warp performance
gpu-algo-test --sizes smaller_than_warp,exactly_one_warp --types floating

# Test block-level algorithms
gpu-algo-test --sizes full_block --types integer

# Test multi-block coordination
gpu-algo-test --sizes several_blocks_boundary,several_blocks_partial --types float,double
```

### Development and Debugging

```bash
# Dry run to check executable discovery
gpu-algo-test --dryrun --verbose

# Test single executable with specific configuration
gpu-algo-test --executables vector_cumsum_parallel --sizes 32 --types float --verbose

# Save detailed results for analysis
gpu-algo-test --output results.json --executables vector_cumsum_parallel
```

### Different Build Configurations

```bash
# Test debug build (default)
gpu-algo-test --preset debug

# Test release build
gpu-algo-test --preset release

# Test from different directory
gpu-algo-test --cmake-root /path/to/project --preset debug

# Use custom binary location
gpu-algo-test --bin-path /custom/path/to/bin
```

## Integration with CMake

The framework automatically integrates with CMake preset configurations. Place your `CMakePresets.json` in the project root:

```json
{
    "version": 3,
    "configurePresets": [
        {
            "name": "debug",
            "binaryDir": "${sourceDir}/builds/debug",
            "cacheVariables": {
                "CMAKE_BUILD_TYPE": "Debug"
            }
        }
    ]
}
```

The framework will automatically find executables in the `bin` subdirectory of the configured binary directory.

## Expected Executable Interface

Executables should support the following command line interface:

```bash
executable --type <data_type> -n <size>
# or
executable --type <data_type> -m <size>
# or
executable --type <data_type> --size <size>
```

Where:
- `<data_type>` is one of: half, float, double, int8, int16, int32, int64, uint8, uint16, uint32, uint64
- `<size>` is the problem size (number of elements)

## Performance Metrics

The framework automatically extracts performance metrics from executable output:

- **Max error**: Numerical accuracy
- **Gross/Net speedup**: Performance comparisons
- **Kernel time**: GPU kernel execution time
- **Total time**: End-to-end execution time

## Development

### Setting up Development Environment

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Run code formatting
black gpu_algo_test/
isort gpu_algo_test/

# Run type checking
mypy gpu_algo_test/

# Run tests (when available)
pytest
```

### Project Structure

```
python/
├── pyproject.toml          # Modern Python project configuration
├── README.md               # This file
├── gpu_algo_test/          # Main package
│   ├── __init__.py         # Package exports
│   └── gpu_algo_test.py    # Core implementation
└── tests/                  # Test suite (optional)
```

## License

Copyright (c) 2025 Alessandro Baretta. All rights reserved.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run the development tools (black, isort, mypy)
5. Submit a pull request

## Support

For issues and questions, please use the [GitHub Issues](https://github.com/alessandrobaretta/gpu-algos/issues) page.
