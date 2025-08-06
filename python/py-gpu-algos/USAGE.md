# py-gpu-algos Usage Guide

This guide provides step-by-step instructions for building, installing, and using the py-gpu-algos Python package.

## Prerequisites

Before building the package, ensure you have:

1. **CUDA Toolkit** (version 11.0 or later)
   - Verify with: `nvcc --version`
   - Ensure `nvcc` is in your PATH

2. **Python 3.8+** with development headers
   - Verify with: `python --version`

3. **Required system packages**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install python3-dev cmake build-essential

   # CentOS/RHEL
   sudo yum install python3-devel cmake gcc-c++
   ```

4. **Required Python packages**:
   ```bash
   pip install numpy pybind11
   ```

5. **Project built at least once**:
   ```bash
   # From project root (gpu_algos/)
   ./scripts/release_build.sh
   ```

## Building and Installing the Package

### Method 1: Integrated Build (Recommended)

This method uses the setup.py that automatically calls the project build scripts:

```bash
# Navigate to the Python package directory
cd python/py-gpu-algos

# Build C++ extensions and install in one step
python setup.py build_ext --release && pip install -e .

# Or for debug build:
python setup.py build_ext --debug && pip install -e .
```

### Method 2: Manual Build + Simple Install

If you prefer to build manually using the main project system:

```bash
# From project root (gpu_algos/)
./scripts/release_build.sh --target py-gpu-algos-modules

# Navigate to Python package directory
cd python/py-gpu-algos

# Copy built modules (if needed)
cp ../../builds/release/python/py-gpu-algos/*.so py_gpu_algos/

# Install package
pip install -e .
```

### Verify Installation

Test that the package works correctly:

```bash
python -c "
import py_gpu_algos
import numpy as np

print('✅ Package imported successfully!')

# Quick test
a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
result = py_gpu_algos.matrix_product_naive(a, b)
print('✅ Matrix multiplication works:', result.flatten())
"
```

Expected output:
```
✅ Package imported successfully!
✅ Matrix multiplication works: [19. 22. 43. 50.]
```

For more comprehensive testing:

```bash
python -c "
import py_gpu_algos
import numpy as np

print('✅ Package imported successfully!')

# Test matrix operations
a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
result = py_gpu_algos.matrix_product_naive(a, b)
print('✅ Matrix multiplication works:', result.flatten())

# Test vector operations
vec = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
cumsum_result = py_gpu_algos.vector_cumsum_serial(vec)
print('✅ Vector cumsum works:', cumsum_result)

# Test GLM operations (requires 3D tensors)
X = np.random.randn(2, 1, 4).astype(np.float32)  # 2 features, 1 task, 4 obs
M = np.random.randn(2, 1, 1).astype(np.float32)  # 2 features, 1 target, 1 task
Yhat = py_gpu_algos.glm_predict_naive(X, M)
print('✅ GLM prediction works, shape:', Yhat.shape)

# Test sort operations (requires power-of-2 dimensions)
tensor = np.random.randint(0, 100, (4, 2, 8), dtype=np.int32)  # 4x2x8 tensor
py_gpu_algos.tensor_sort_bitonic(tensor, 'rows')  # Sort along first dim (size 4 = 2^2)
print('✅ Tensor sort works!')

print('✅ All operations successful!')
"
```

## Usage Examples

### Matrix Operations

```python
import numpy as np
import py_gpu_algos

# Basic matrix multiplication
a = np.random.randn(100, 50).astype(np.float32)
b = np.random.randn(50, 80).astype(np.float32)

# GPU computation using high-level interface
c_gpu = py_gpu_algos.matrix_product_naive(a, b)
c_tiled = py_gpu_algos.matrix_product_tiled(a, b)  # Tiled algorithm

# Compare with NumPy
c_numpy = np.dot(a, b)
print(f"Naive results match: {np.allclose(c_gpu, c_numpy)}")
print(f"Tiled results match: {np.allclose(c_tiled, c_numpy)}")

# Matrix transpose
matrix = np.random.randn(64, 128).astype(np.float32)
transposed = py_gpu_algos.matrix_transpose_striped(matrix)
print(f"Transpose shape: {matrix.shape} -> {transposed.shape}")
```

### Vector Operations

```python
import numpy as np
import py_gpu_algos

# Cumulative sum operations
vec = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)

# Serial cumulative sum
cumsum_serial = py_gpu_algos.vector_cumsum_serial(vec)
print(f"Serial cumsum: {cumsum_serial}")  # [1, 3, 6, 10, 15]

# Parallel cumulative sum
cumsum_parallel = py_gpu_algos.vector_cumsum_parallel(vec)
print(f"Parallel cumsum: {cumsum_parallel}")

# Cumulative maximum
cummax = py_gpu_algos.vector_cummax_parallel(vec)
print(f"Cummax: {cummax}")  # [1, 2, 3, 4, 5]

# Generic scan operations
scan_sum = py_gpu_algos.vector_scan_parallel(vec, "sum")
scan_max = py_gpu_algos.vector_scan_parallel(vec, "max")
scan_min = py_gpu_algos.vector_scan_parallel(vec, "min")
scan_prod = py_gpu_algos.vector_scan_parallel(vec, "prod")

print(f"Scan sum: {scan_sum}")
print(f"Scan max: {scan_max}")
print(f"Scan min: {scan_min}")
print(f"Scan prod: {scan_prod}")
```

### GLM Operations (3D Tensor Operations)

```python
import numpy as np
import py_gpu_algos

# GLM operations work with 3D tensors for multitask learning
nfeatures = 10
ntargets = 3
ntasks = 5
nobs = 100

# Create input tensors
X = np.random.randn(nfeatures, ntasks, nobs).astype(np.float32)  # Features
Y = np.random.randn(ntargets, ntasks, nobs).astype(np.float32)   # Targets
M = np.random.randn(nfeatures, ntargets, ntasks).astype(np.float32)  # Model

# Prediction: Ŷ = X^T * M
Yhat = py_gpu_algos.glm_predict_naive(X, M)
print(f"Prediction shape: {Yhat.shape}")  # (ntargets, ntasks, nobs)

# Gradient computation for linear regression
grad_naive = py_gpu_algos.glm_gradient_naive(X, Y, M)
grad_optimized = py_gpu_algos.glm_gradient_xyyhat(X, Y, M)

print(f"Gradient shape: {grad_naive.shape}")  # (nfeatures, ntargets, ntasks)
print(f"Gradients match: {np.allclose(grad_naive, grad_optimized)}")

# Example: Simple linear regression for one task
print("\n--- Simple Linear Regression Example ---")
X_simple = np.random.randn(3, 1, 50).astype(np.float32)  # 3 features, 1 task, 50 obs
M_simple = np.array([[[0.5]], [[1.2]], [[-0.3]]], dtype=np.float32)  # 3x1x1 model
Y_true = py_gpu_algos.glm_predict_naive(X_simple, M_simple)

# Add noise
Y_noisy = Y_true + 0.1 * np.random.randn(*Y_true.shape).astype(np.float32)

# Compute gradient for parameter update
gradient = py_gpu_algos.glm_gradient_naive(X_simple, Y_noisy, M_simple)
print(f"Gradient for parameter update: {gradient.flatten()}")
```

### Sort Operations (3D Tensor Sorting)

```python
import numpy as np
import py_gpu_algos

# Bitonic sort requires power-of-2 dimensions
# Create a 3D tensor with power-of-2 sizes
tensor = np.random.randint(0, 100, (8, 4, 16), dtype=np.int32)  # 8x4x16
print(f"Original tensor shape: {tensor.shape}")
print(f"Sample before sorting:\n{tensor[0, 0, :8]}")  # First 8 elements

# Sort along different dimensions (in-place operation)
tensor_copy1 = tensor.copy()
py_gpu_algos.tensor_sort_bitonic(tensor_copy1, "rows")  # Sort along dim 0 (size 8)
print(f"After sorting rows: {tensor_copy1[0, 0, :8]}")

tensor_copy2 = tensor.copy()
py_gpu_algos.tensor_sort_bitonic(tensor_copy2, "cols")  # Sort along dim 1 (size 4)

tensor_copy3 = tensor.copy()
py_gpu_algos.tensor_sort_bitonic(tensor_copy3, "sheets")  # Sort along dim 2 (size 16)

# Verify sorting worked
print(f"Rows sorted: {np.all(tensor_copy1[0, 0, :] == np.sort(tensor[0, 0, :]))}")

# Note: Sort operations modify the tensor in-place
print("\n⚠️  Important: tensor_sort_bitonic modifies the input tensor in-place!")

# Float sorting example
float_tensor = np.random.randn(4, 8, 2).astype(np.float32)
original = float_tensor.copy()
py_gpu_algos.tensor_sort_bitonic(float_tensor, "cols")  # Sort along cols (size 8)
print(f"Float tensor sorted along cols: {np.allclose(float_tensor[0, :, 0], np.sort(original[0, :, 0]))}")
```

### Type-Specific Functions

```python
import numpy as np
import py_gpu_algos

# Float32 matrices
a_f32 = np.random.randn(64, 64).astype(np.float32)
b_f32 = np.random.randn(64, 64).astype(np.float32)
c_f32 = py_gpu_algos.matrix_product_naive_float32(a_f32, b_f32)

# Float64 matrices
a_f64 = np.random.randn(32, 32).astype(np.float64)
b_f64 = np.random.randn(32, 32).astype(np.float64)
c_f64 = py_gpu_algos.matrix_product_naive_float64(a_f64, b_f64)

# Int32 matrices
a_i32 = np.random.randint(-10, 10, (32, 32), dtype=np.int32)
b_i32 = np.random.randint(-10, 10, (32, 32), dtype=np.int32)
c_i32 = py_gpu_algos.matrix_product_naive_int32(a_i32, b_i32)

print("All computations completed successfully!")
```

### Different Import Styles

```python
# Import entire package
import py_gpu_algos

# Matrix operations
a = np.random.randn(64, 64).astype(np.float32)
b = np.random.randn(64, 64).astype(np.float32)
result = py_gpu_algos.matrix_product_naive(a, b)

# Vector operations
vec = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
cumsum = py_gpu_algos.vector_cumsum_serial(vec)

# GLM operations
X = np.random.randn(5, 2, 10).astype(np.float32)
M = np.random.randn(5, 1, 2).astype(np.float32)
predictions = py_gpu_algos.glm_predict_naive(X, M)

# Sort operations
tensor = np.random.randint(0, 100, (8, 4, 16), dtype=np.int32)
py_gpu_algos.tensor_sort_bitonic(tensor, "rows")

# Import specific functions
from py_gpu_algos import (
    matrix_product_naive, matrix_product_tiled,
    vector_cumsum_serial, vector_scan_parallel,
    glm_predict_naive, tensor_sort_bitonic
)

result1 = matrix_product_naive(a, b)
result2 = matrix_product_tiled(a, b)
cumsum = vector_cumsum_serial(vec)
scan_result = vector_scan_parallel(vec, "sum")
predictions = glm_predict_naive(X, M)
tensor_sort_bitonic(tensor, "cols")

# Import specific modules
from py_gpu_algos import matrix_ops, vector_ops, glm_ops, sort_ops

# Use module-specific functions
matrix_result = matrix_ops.matrix_product_naive(a, b)
vector_result = vector_ops.vector_cumsum_parallel(vec)
glm_result = glm_ops.glm_gradient_naive(X, Y, M)
sort_ops.tensor_sort_bitonic(tensor, "sheets")

# Import type-specific functions for performance-critical code
from py_gpu_algos import (
    matrix_product_naive_float32,
    vector_cumsum_serial_float32,
    glm_predict_naive_float32,
    tensor_sort_bitonic_float32
)

# Use with guaranteed float32 types (no runtime dispatch overhead)
result_f32 = matrix_product_naive_float32(a, b)
cumsum_f32 = vector_cumsum_serial_float32(vec)
pred_f32 = glm_predict_naive_float32(X, M)
tensor_sort_bitonic_float32(tensor.astype(np.float32), "rows")
```

### Performance Comparison

```python
import time
import numpy as np
import py_gpu_algos

def benchmark_matrix_multiply(size=512):
    """Benchmark matrix multiplication operations."""
    # Create test matrices
    a = np.random.randn(size, size).astype(np.float32)
    b = np.random.randn(size, size).astype(np.float32)

    # Warm up GPU
    _ = py_gpu_algos.matrix_product_naive(a[:10, :10], b[:10, :10])

    # NumPy timing
    start = time.time()
    c_numpy = np.dot(a, b)
    numpy_time = time.time() - start

    # GPU naive timing
    start = time.time()
    c_gpu_naive = py_gpu_algos.matrix_product_naive(a, b)
    gpu_naive_time = time.time() - start

    # GPU tiled timing
    start = time.time()
    c_gpu_tiled = py_gpu_algos.matrix_product_tiled(a, b)
    gpu_tiled_time = time.time() - start

    # Verify correctness
    naive_error = np.max(np.abs(c_gpu_naive - c_numpy))
    tiled_error = np.max(np.abs(c_gpu_tiled - c_numpy))

    print(f"Matrix Multiplication Benchmark ({size}x{size}):")
    print(f"  NumPy time:    {numpy_time*1000:.2f} ms")
    print(f"  GPU naive:     {gpu_naive_time*1000:.2f} ms (speedup: {numpy_time/gpu_naive_time:.2f}x)")
    print(f"  GPU tiled:     {gpu_tiled_time*1000:.2f} ms (speedup: {numpy_time/gpu_tiled_time:.2f}x)")
    print(f"  Naive error:   {naive_error:.2e}")
    print(f"  Tiled error:   {tiled_error:.2e}")

def benchmark_vector_operations(size=1000000):
    """Benchmark vector operations."""
    vec = np.random.randn(size).astype(np.float32)

    # Warm up
    _ = py_gpu_algos.vector_cumsum_serial(vec[:100])

    # NumPy cumsum
    start = time.time()
    numpy_cumsum = np.cumsum(vec)
    numpy_time = time.time() - start

    # GPU serial cumsum
    start = time.time()
    gpu_serial = py_gpu_algos.vector_cumsum_serial(vec)
    gpu_serial_time = time.time() - start

    # GPU parallel cumsum
    start = time.time()
    gpu_parallel = py_gpu_algos.vector_cumsum_parallel(vec)
    gpu_parallel_time = time.time() - start

    print(f"\nVector Cumsum Benchmark (size: {size}):")
    print(f"  NumPy time:        {numpy_time*1000:.2f} ms")
    print(f"  GPU serial:        {gpu_serial_time*1000:.2f} ms (speedup: {numpy_time/gpu_serial_time:.2f}x)")
    print(f"  GPU parallel:      {gpu_parallel_time*1000:.2f} ms (speedup: {numpy_time/gpu_parallel_time:.2f}x)")
    print(f"  Serial error:      {np.max(np.abs(gpu_serial - numpy_cumsum)):.2e}")
    print(f"  Parallel error:    {np.max(np.abs(gpu_parallel - numpy_cumsum)):.2e}")

def benchmark_glm_operations():
    """Benchmark GLM operations."""
    nfeatures, ntargets, ntasks, nobs = 100, 10, 20, 1000

    X = np.random.randn(nfeatures, ntasks, nobs).astype(np.float32)
    Y = np.random.randn(ntargets, ntasks, nobs).astype(np.float32)
    M = np.random.randn(nfeatures, ntargets, ntasks).astype(np.float32)

    # Warm up
    _ = py_gpu_algos.glm_predict_naive(X[:10, :5, :10], M[:10, :5, :5])

    # GLM prediction timing
    start = time.time()
    Yhat = py_gpu_algos.glm_predict_naive(X, M)
    predict_time = time.time() - start

    # GLM gradient timing (naive vs optimized)
    start = time.time()
    grad_naive = py_gpu_algos.glm_gradient_naive(X, Y, M)
    grad_naive_time = time.time() - start

    start = time.time()
    grad_optimized = py_gpu_algos.glm_gradient_xyyhat(X, Y, M)
    grad_optimized_time = time.time() - start

    print(f"\nGLM Operations Benchmark ({nfeatures}f, {ntargets}t, {ntasks}T, {nobs}obs):")
    print(f"  Prediction:        {predict_time*1000:.2f} ms")
    print(f"  Gradient (naive):  {grad_naive_time*1000:.2f} ms")
    print(f"  Gradient (opt):    {grad_optimized_time*1000:.2f} ms (speedup: {grad_naive_time/grad_optimized_time:.2f}x)")
    print(f"  Gradient error:    {np.max(np.abs(grad_naive - grad_optimized)):.2e}")

def benchmark_sort_operations():
    """Benchmark sort operations."""
    # Use power-of-2 dimensions for bitonic sort
    tensor = np.random.randint(0, 10000, (64, 32, 128), dtype=np.int32)

    # Create copies for different sort dimensions
    tensor_rows = tensor.copy()
    tensor_cols = tensor.copy()
    tensor_sheets = tensor.copy()

    # Sort along rows (dim 0, size 64 = 2^6)
    start = time.time()
    py_gpu_algos.tensor_sort_bitonic(tensor_rows, "rows")
    sort_rows_time = time.time() - start

    # Sort along cols (dim 1, size 32 = 2^5)
    start = time.time()
    py_gpu_algos.tensor_sort_bitonic(tensor_cols, "cols")
    sort_cols_time = time.time() - start

    # Sort along sheets (dim 2, size 128 = 2^7)
    start = time.time()
    py_gpu_algos.tensor_sort_bitonic(tensor_sheets, "sheets")
    sort_sheets_time = time.time() - start

    print(f"\nTensor Sort Benchmark (64x32x128):")
    print(f"  Sort rows (64):    {sort_rows_time*1000:.2f} ms")
    print(f"  Sort cols (32):    {sort_cols_time*1000:.2f} ms")
    print(f"  Sort sheets (128): {sort_sheets_time*1000:.2f} ms")

# Run all benchmarks
if __name__ == "__main__":
    benchmark_matrix_multiply(512)
    benchmark_vector_operations(1000000)
    benchmark_glm_operations()
    benchmark_sort_operations()
```

## Available Functions

### Matrix Operations

**High-Level Interface (Automatic Type Dispatch):**
- `py_gpu_algos.matrix_product_naive(a, b)` - Naive matrix multiplication
- `py_gpu_algos.matrix_product_tiled(a, b)` - Tiled matrix multiplication
- `py_gpu_algos.matrix_transpose_striped(a)` - Matrix transpose

**Low-Level Type-Specific Interface:**
All matrix operations support the full range of types: `float32`, `float64`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`

Examples:
- `py_gpu_algos.matrix_product_naive_float32(a, b)`
- `py_gpu_algos.matrix_product_tiled_float64(a, b)`
- `py_gpu_algos.matrix_transpose_striped_int32(a)`

### Vector Operations

**High-Level Interface:**
- `py_gpu_algos.vector_cumsum_serial(a)` - Serial cumulative sum
- `py_gpu_algos.vector_cumsum_parallel(a)` - Parallel cumulative sum
- `py_gpu_algos.vector_cummax_parallel(a)` - Parallel cumulative maximum
- `py_gpu_algos.vector_scan_parallel(a, operation)` - Generic parallel scan
  - `operation` can be: `"sum"`, `"max"`, `"min"`, `"prod"`

**Low-Level Type-Specific Interface:**
All vector operations support: `float32`, `float64`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`

Examples:
- `py_gpu_algos.vector_cumsum_serial_float32(a)`
- `py_gpu_algos.vector_scan_parallel_sum_int32(a)`
- `py_gpu_algos.vector_scan_parallel_max_float64(a)`

### GLM Operations (3D Tensors)

**High-Level Interface:**
- `py_gpu_algos.glm_predict_naive(X, M)` - Linear model prediction
- `py_gpu_algos.glm_gradient_naive(X, Y, M)` - Gradient computation (naive)
- `py_gpu_algos.glm_gradient_xyyhat(X, Y, M)` - Gradient computation (optimized)

**Low-Level Type-Specific Interface:**
GLM operations support: `float32`, `float64`, `int32`, `int64`

Examples:
- `py_gpu_algos.glm_predict_naive_float32(X, M)`
- `py_gpu_algos.glm_gradient_naive_float64(X, Y, M)`
- `py_gpu_algos.glm_gradient_xyyhat_int32(X, Y, M)`

### Sort Operations (3D Tensors)

**High-Level Interface:**
- `py_gpu_algos.tensor_sort_bitonic(tensor, sort_dim)` - In-place bitonic sort
  - `sort_dim` can be: `"rows"`, `"cols"`, `"sheets"`
  - ⚠️ **Requirement:** Sort dimension size must be a power of 2

**Low-Level Type-Specific Interface:**
Sort operations support all numeric types: `float32`, `float64`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`

Examples:
- `py_gpu_algos.tensor_sort_bitonic_float32(tensor, sort_dim)`
- `py_gpu_algos.tensor_sort_bitonic_int64(tensor, sort_dim)`

### Function Signatures

**Matrix Operations:**
```python
def matrix_product_naive(a: NDArray[T], b: NDArray[T]) -> NDArray[T]:
    """
    Compute matrix product C = A * B using naive algorithm on GPU.

    Args:
        a: Input matrix A of shape (m, k)
        b: Input matrix B of shape (k, n)

    Returns:
        Result matrix C of shape (m, n)
    """

def matrix_transpose_striped(a: NDArray[T]) -> NDArray[T]:
    """
    Compute matrix transpose using striped algorithm on GPU.

    Args:
        a: Input matrix of shape (m, n)

    Returns:
        Transposed matrix of shape (n, m)
    """
```

**Vector Operations:**
```python
def vector_cumsum_serial(a: NDArray[T]) -> NDArray[T]:
    """
    Compute cumulative sum using serial algorithm.

    Args:
        a: Input vector of shape (n,)

    Returns:
        Cumulative sum vector of shape (n,)
    """

def vector_scan_parallel(a: NDArray[T], operation: str) -> NDArray[T]:
    """
    Compute parallel scan with specified operation.

    Args:
        a: Input vector of shape (n,)
        operation: One of "sum", "max", "min", "prod"

    Returns:
        Scan result vector of shape (n,)
    """
```

**GLM Operations:**
```python
def glm_predict_naive(X: NDArray[T], M: NDArray[T]) -> NDArray[T]:
    """
    Compute GLM predictions for multitask learning.

    Args:
        X: Features tensor of shape (nfeatures, ntasks, nobs)
        M: Model tensor of shape (nfeatures, ntargets, ntasks)

    Returns:
        Predictions tensor of shape (ntargets, ntasks, nobs)
    """

def glm_gradient_naive(X: NDArray[T], Y: NDArray[T], M: NDArray[T]) -> NDArray[T]:
    """
    Compute GLM gradient for linear regression.

    Args:
        X: Features tensor of shape (nfeatures, ntasks, nobs)
        Y: Targets tensor of shape (ntargets, ntasks, nobs)
        M: Model tensor of shape (nfeatures, ntargets, ntasks)

    Returns:
        Gradient tensor of shape (nfeatures, ntargets, ntasks)
    """
```

**Sort Operations:**
```python
def tensor_sort_bitonic(tensor: NDArray[T], sort_dim: str) -> None:
    """
    Perform in-place bitonic sort on 3D tensor.

    Args:
        tensor: Input 3D tensor (modified in-place)
        sort_dim: Dimension to sort ("rows", "cols", "sheets")

    Returns:
        None (modifies input tensor in-place)

    Raises:
        ValueError: If sort dimension size is not a power of 2
    """
```

**Common Error Conditions:**
All functions raise `ValueError` for:
- Incompatible array shapes
- Mismatched dtypes between inputs
- Unsupported dtypes for the operation
- Non-contiguous arrays (automatically fixed with `ascontiguousarray`)

## Error Handling

### Common Errors and Solutions

```python
import numpy as np
import py_gpu_algos

# 1. Matrix dimension mismatch
try:
    a = np.random.randn(10, 5).astype(np.float32)
    b = np.random.randn(8, 12).astype(np.float32)  # Wrong: 5 != 8
    c = py_gpu_algos.matrix_product_naive(a, b)
except ValueError as e:
    print(f"Matrix dimension error: {e}")

# 2. Type mismatch
try:
    a = np.random.randn(10, 5).astype(np.float32)
    b = np.random.randn(5, 8).astype(np.float64)   # Wrong: float32 != float64
    c = py_gpu_algos.matrix_product_naive(a, b)
except ValueError as e:
    print(f"Type error: {e}")

# 3. Non-contiguous arrays
try:
    a = np.random.randn(10, 10).astype(np.float32)
    a_view = a[::2, ::2]  # Non-contiguous view
    b = np.random.randn(5, 8).astype(np.float32)
    c = py_gpu_algos.matrix_product_naive(a_view, b)
except ValueError as e:
    print(f"Contiguity error: {e}")

    # Fix: make contiguous
    a_contiguous = np.ascontiguousarray(a_view)
    c = py_gpu_algos.matrix_product_naive(a_contiguous, b)
    print("Fixed with ascontiguousarray!")

# 4. GLM tensor dimension mismatch
try:
    X = np.random.randn(10, 5, 100).astype(np.float32)  # 10 features, 5 tasks, 100 obs
    M = np.random.randn(8, 3, 5).astype(np.float32)    # Wrong: 10 != 8 features
    Yhat = py_gpu_algos.glm_predict_naive(X, M)
except ValueError as e:
    print(f"GLM dimension error: {e}")

# 5. GLM unsupported dtype
try:
    X = np.random.randn(5, 2, 10).astype(np.int8)  # GLM doesn't support int8
    M = np.random.randn(5, 1, 2).astype(np.int8)
    Yhat = py_gpu_algos.glm_predict_naive(X, M)
except ValueError as e:
    print(f"GLM dtype error: {e}")

# 6. Sort operation with non-power-of-2 dimension
try:
    tensor = np.random.randint(0, 100, (7, 4, 8), dtype=np.int32)  # 7 is not power of 2
    py_gpu_algos.tensor_sort_bitonic(tensor, "rows")
except ValueError as e:
    print(f"Sort dimension error: {e}")

    # Fix: use power-of-2 dimensions
    tensor_fixed = np.random.randint(0, 100, (8, 4, 8), dtype=np.int32)  # 8 = 2^3
    py_gpu_algos.tensor_sort_bitonic(tensor_fixed, "rows")
    print("Fixed with power-of-2 dimension!")

# 7. Wrong number of arguments for vector operations
try:
    vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = py_gpu_algos.vector_cumsum_serial(vec, "extra_arg")  # Wrong: too many args
except TypeError as e:
    print(f"Vector function error: {e}")

# 8. Invalid scan operation
try:
    vec = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    result = py_gpu_algos.vector_scan_parallel(vec, "invalid_op")  # Wrong operation
except ValueError as e:
    print(f"Scan operation error: {e}")

    # Fix: use valid operations
    result = py_gpu_algos.vector_scan_parallel(vec, "sum")  # Valid: sum, max, min, prod
    print("Fixed with valid operation!")
```

## Troubleshooting

### Build Issues

**Problem: "No module named 'pybind11'"**
```bash
pip install pybind11
```

**Problem: "nvcc not found"**
```bash
# Add CUDA to PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**Problem: "Build script not found"**
```bash
# Ensure you're in the right directory
ls scripts/release_build.sh  # Should exist from project root
```

**Problem: CMake configuration errors**
```bash
# Clean and reconfigure
rm -rf builds/
./scripts/release_build.sh
```

### Installation Issues

**Problem: "pip install fails"**
```bash
# Try upgrading setuptools
pip install --upgrade setuptools wheel

# Or use legacy install
pip install -e . --config-settings editable_mode=compat
```

**Problem: "Package not found after install"**
```bash
# Check installation
pip list | grep py-gpu-algos

# Uninstall and reinstall if needed
pip uninstall py-gpu-algos -y
pip install -e .
```

### Runtime Issues

**Problem: "ImportError: attempted relative import"**
```bash
# Make sure you're importing the package, not the module directly
python -c "import py_gpu_algos"  # Correct
python -c "import matrix_ops"    # Wrong
```

**Problem: "CUDA runtime errors"**
```bash
# Check GPU availability
nvidia-smi

# Check CUDA installation
python -c "
import py_gpu_algos
from py_gpu_algos import matrix_ops
print('CUDA available:', matrix_ops._CUDA_AVAILABLE)
"
```

**Problem: "Memory errors with large matrices"**
```python
# Monitor GPU memory
import subprocess
subprocess.run(['nvidia-smi'])

# Use smaller matrices or free GPU memory
import gc
gc.collect()
```

### Getting Help

1. **Verify prerequisites**: Ensure CUDA, Python, and required packages are installed
2. **Check build output**: Look for specific error messages during build
3. **Test incrementally**: Start with small matrices and simple examples
4. **Clean rebuild**: Remove `builds/` directory and rebuild from scratch

## Development Notes

This package provides comprehensive Python bindings for the **entire** GPU algorithms library:

### Currently Implemented Operations

**Matrix Operations (2D):**
- **Naive matrix multiplication**: O(n³) algorithm optimized for GPU
- **Tiled matrix multiplication**: Memory-efficient blocked algorithm
- **Matrix transpose**: Striped algorithm for optimal memory access

**Vector Operations (1D):**
- **Cumulative sum**: Both serial and parallel implementations
- **Cumulative maximum**: Parallel implementation
- **Generic scan operations**: Sum, max, min, product with parallel algorithms

**GLM Operations (3D Tensors):**
- **GLM prediction**: Linear model prediction for multitask learning
- **GLM gradients**: Both naive and optimized gradient computation algorithms
- **Multitask support**: Handle multiple tasks and targets simultaneously

**Sort Operations (3D Tensors):**
- **Bitonic sort**: In-place sorting for 3D tensors (requires power-of-2 dimensions)
- **Multi-dimensional**: Sort along any dimension (rows, cols, sheets)

### Technical Features

- **Comprehensive type support**: 11 different NumPy dtypes (float32/64, int8-64, uint8-64)
- **Two-tier API**: Low-level type-specific + high-level automatic dispatch functions
- **Memory safety**: Automatic GPU memory management and comprehensive error checking
- **Efficient data handling**: Minimal copying with automatic contiguity handling
- **CUDA streams**: Asynchronous GPU execution with proper synchronization
- **Error propagation**: Detailed error messages with context

### Performance Characteristics

- **Matrix operations**: Competitive with optimized BLAS libraries for large matrices
- **Vector operations**: Significant speedups for large vectors (>100K elements)
- **GLM operations**: Optimized for multitask learning workloads
- **Sort operations**: GPU-accelerated bitonic sort for structured 3D data

### Architecture Design

**Binding Layer (C++):**
- Pybind11-based bindings with template instantiation
- Direct CUDA kernel integration
- Comprehensive input validation and error handling

**Python Layer:**
- NumPy integration with type preservation
- Overloaded functions for seamless user experience
- Automatic type dispatch with fallback support

**Build System:**
- CMake integration with parent project
- Automated CUDA compilation and linking
- Cross-platform compatibility (Linux primary, Windows/macOS planned)

This represents a **complete implementation** of all available GPU algorithms in the parent C++ library, providing Python users with full access to optimized GPU computing capabilities.
