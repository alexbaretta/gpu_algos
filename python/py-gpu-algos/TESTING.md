# Testing Guide for py-gpu-algos

This guide provides instructions for running the pytest test suite for py-gpu-algos.

## Quick Start

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test categories
pytest -m "not slow"          # Skip slow tests
pytest -m performance         # Performance tests only
pytest -m error_handling      # Error handling tests only
```

## Test Categories

The test suite uses markers to organize tests:

| Marker           | Description               | Usage                      |
|------------------|---------------------------|----------------------------|
| `slow`           | Time-intensive tests      | `pytest -m "not slow"`     |
| `performance`    | GPU vs CPU benchmarks     | `pytest -m performance`    |
| `integration`    | Multi-operation workflows | `pytest -m integration`    |
| `error_handling` | Error conditions          | `pytest -m error_handling` |

## Running Specific Tests

```bash
# Test specific operations
pytest tests/test_matrix_ops.py
pytest tests/test_vector_ops.py
pytest tests/test_glm_ops.py
pytest tests/test_sort_ops.py

# Test specific functions
pytest tests/test_matrix_ops.py::test_matrix_product_naive
pytest -k "cumsum"            # All cumsum-related tests
pytest -k "float32"           # All float32 tests
```

## Test Runner (Alternative)

Use the provided test runner for organized execution:

```bash
# Quick validation (excludes slow tests)
python run_tests.py --category fast

# Performance benchmarks
python run_tests.py --category performance

# Error handling validation
python run_tests.py --category error-handling

# Full test suite
python run_tests.py --category integration
```

## Configuration

Tests are configured via `tests/pytest.ini`:
- Automatic test discovery
- Verbose output by default
- Strict marker enforcement

## Requirements

- CUDA-capable GPU
- Python 3.8+
- NumPy, pytest
- py-gpu-algos package installed

## Troubleshooting

**CUDA errors**: Ensure GPU drivers and CUDA toolkit are properly installed
**Import errors**: Verify py-gpu-algos is installed: `pip install -e .`
**Slow performance**: Run without performance tests: `pytest -m "not performance"`
