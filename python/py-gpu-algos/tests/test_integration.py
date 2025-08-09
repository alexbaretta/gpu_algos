"""
Integration tests for py-gpu-algos.

Tests complex workflows combining multiple operations across different modules:
- Matrix-vector operation chains
- GLM workflow simulations
- Multi-step computational pipelines
- Cross-module data flow and compatibility
- Real-world usage patterns
"""

import pytest
import numpy as np
import py_gpu_algos
from .utils import (
    assert_array_close, validate_basic_properties,
    get_numpy_reference_matrix_product, get_numpy_reference_matrix_transpose,
    get_numpy_reference_cumsum_parallel, get_numpy_reference_glm_predict,
    get_numpy_reference_glm_gradient, get_numpy_reference_tensor_sort
)

class TestMatrixVectorIntegration:
    """Test integration between matrix and vector operations."""

    def test_matrix_product_then_vector_ops(self, dtype_float):
        """Test matrix multiplication followed by vector operations on result."""
        # Create matrices
        np.random.seed(42)
        a = np.random.randn(64, 48).astype(dtype_float)
        b = np.random.randn(48, 1).astype(dtype_float)  # Column vector as matrix

        # Matrix multiplication to get column vector result
        matrix_result = py_gpu_algos.matrix_product_naive(a, b)
        assert matrix_result.shape == (64, 1)

        # Convert to 1D vector for vector operations
        vector_result = matrix_result.flatten()

        # Apply vector operations
        cumsum_result = py_gpu_algos.vector_cumsum_parallel(vector_result)
        cummax_result = py_gpu_algos.vector_cummax_parallel(vector_result)

        # Verify with NumPy reference
        numpy_matrix = get_numpy_reference_matrix_product(a, b)
        numpy_vector = numpy_matrix.flatten()
        numpy_cumsum = get_numpy_reference_cumsum_parallel(numpy_vector)
        numpy_cummax = np.maximum.accumulate(numpy_vector)

        assert_array_close(cumsum_result, numpy_cumsum, dtype_float)
        assert_array_close(cummax_result, numpy_cummax, dtype_float)

    def test_vector_scan_then_matrix_reshape(self, dtype_float):
        """Test vector scan followed by reshaping and matrix operations."""
        # Create a vector that can be reshaped into a square matrix
        size = 64
        vector = np.random.randn(size * size).astype(dtype_float)

        # Apply vector scan
        scan_result = py_gpu_algos.vector_scan_parallel(vector, "sum")

        # Reshape to matrix
        matrix = scan_result.reshape(size, size)

        # Apply matrix transpose
        transposed = py_gpu_algos.matrix_transpose_striped(matrix)

        # Verify with NumPy reference
        numpy_scan = np.cumsum(vector)
        numpy_matrix = numpy_scan.reshape(size, size)
        numpy_transposed = numpy_matrix.T

        assert_array_close(transposed, numpy_transposed, dtype_float)

    def test_matrix_operations_chain(self, dtype_float):
        """Test chaining multiple matrix operations."""
        # Create matrices
        np.random.seed(42)
        a = np.random.randn(32, 24).astype(dtype_float)
        b = np.random.randn(24, 40).astype(dtype_float)
        c = np.random.randn(40, 16).astype(dtype_float)

        # Chain: A @ B, then transpose, then @ C
        ab = py_gpu_algos.matrix_product_naive(a, b)
        ab_t = py_gpu_algos.matrix_transpose_striped(ab)
        result = py_gpu_algos.matrix_product_naive(ab_t, c)

        # Verify with NumPy
        numpy_ab = get_numpy_reference_matrix_product(a, b)
        numpy_ab_t = get_numpy_reference_matrix_transpose(numpy_ab)
        numpy_result = get_numpy_reference_matrix_product(numpy_ab_t, c)

        assert_array_close(result, numpy_result, dtype_float)

class TestGLMWorkflowIntegration:
    """Test complete GLM workflow integration."""

    def test_complete_glm_training_step(self, dtype_glm):
        """Test a complete GLM training step: predict -> gradient -> update."""
        if dtype_glm not in [np.float32, np.float64]:
            pytest.skip("Training simulation only for floating point")

        np.random.seed(42)
        nfeatures, ntargets, ntasks, nobs = 20, 5, 3, 200

        # Generate training data
        X = np.random.randn(nfeatures, ntasks, nobs).astype(dtype_glm)
        Y_true = np.random.randn(ntargets, ntasks, nobs).astype(dtype_glm)

        # Initialize model parameters
        M = np.random.randn(nfeatures, ntargets, ntasks).astype(dtype_glm) * 0.1

        # Training step 1: Forward pass
        Y_pred = py_gpu_algos.glm_predict_naive(X, M)

        # Compute initial loss
        loss_initial = np.mean((Y_pred - Y_true) ** 2)

        # Training step 2: Backward pass
        gradient = py_gpu_algos.glm_gradient_naive(X, Y_true, M)

        # Training step 3: Parameter update
        learning_rate = 0.001
        M_updated = M - learning_rate * gradient

        # Training step 4: Forward pass with updated parameters
        Y_pred_updated = py_gpu_algos.glm_predict_naive(X, M_updated)
        loss_updated = np.mean((Y_pred_updated - Y_true) ** 2)

        # Loss should decrease
        assert loss_updated < loss_initial, f"Loss should decrease: {loss_initial} -> {loss_updated}"

        # Verify against NumPy reference
        numpy_pred_initial = get_numpy_reference_glm_predict(X, M)
        numpy_gradient = get_numpy_reference_glm_gradient(X, Y_true, M)
        numpy_M_updated = M - learning_rate * numpy_gradient
        numpy_pred_updated = get_numpy_reference_glm_predict(X, numpy_M_updated)

        assert_array_close(Y_pred, numpy_pred_initial, dtype_glm)
        assert_array_close(gradient, numpy_gradient, dtype_glm)
        assert_array_close(Y_pred_updated, numpy_pred_updated, dtype_glm)

    def test_glm_multi_algorithm_consistency(self, dtype_glm):
        """Test consistency between different GLM gradient algorithms."""
        np.random.seed(42)
        nfeatures, ntargets, ntasks, nobs = 12, 4, 2, 150

        X = np.random.randn(nfeatures, ntasks, nobs).astype(dtype_glm)
        Y = np.random.randn(ntargets, ntasks, nobs).astype(dtype_glm)
        M = np.random.randn(nfeatures, ntargets, ntasks).astype(dtype_glm)

        if np.issubdtype(dtype_glm, np.integer):
            scale = 5
            X = (X * scale).astype(dtype_glm)
            Y = (Y * scale).astype(dtype_glm)
            M = (M * scale).astype(dtype_glm)

        # Compute gradients with both algorithms
        grad_naive = py_gpu_algos.glm_gradient_naive(X, Y, M)
        grad_xyyhat = py_gpu_algos.glm_gradient_xyyhat(X, Y, M)

        # Both should give the same result
        if dtype_glm in [np.float32, np.float64]:
            rtol = 1e-4 if dtype_glm == np.float32 else 1e-10
            assert_array_close(grad_naive, grad_xyyhat, dtype_glm, rtol=rtol)
        else:
            assert_array_close(grad_naive, grad_xyyhat, dtype_glm)

        # Use both gradients for parameter updates
        learning_rate = 0.01 if dtype_glm in [np.float32, np.float64] else 0.1

        M_updated_naive = M - learning_rate * grad_naive
        M_updated_xyyhat = M - learning_rate * grad_xyyhat

        # Updated parameters should be identical
        assert_array_close(M_updated_naive, M_updated_xyyhat, dtype_glm, rtol=rtol if dtype_glm in [np.float32, np.float64] else 0)

    def test_glm_batch_processing_simulation(self, dtype_glm):
        """Test GLM operations with multiple batches."""
        np.random.seed(42)
        nfeatures, ntargets, ntasks = 15, 6, 4
        batch_sizes = [50, 100, 150]

        X_batches = []
        Y_batches = []
        for batch_size in batch_sizes:
            X_batch = np.random.randn(nfeatures, ntasks, batch_size).astype(dtype_glm)
            Y_batch = np.random.randn(ntargets, ntasks, batch_size).astype(dtype_glm)

            if np.issubdtype(dtype_glm, np.integer):
                scale = 8
                X_batch = (X_batch * scale).astype(dtype_glm)
                Y_batch = (Y_batch * scale).astype(dtype_glm)

            X_batches.append(X_batch)
            Y_batches.append(Y_batch)

        # Shared model parameters
        M = np.random.randn(nfeatures, ntargets, ntasks).astype(dtype_glm)
        if np.issubdtype(dtype_glm, np.integer):
            M = (M * 5).astype(dtype_glm)

        # Process each batch
        predictions = []
        gradients = []

        for X_batch, Y_batch in zip(X_batches, Y_batches):
            pred = py_gpu_algos.glm_predict_naive(X_batch, M)
            grad = py_gpu_algos.glm_gradient_naive(X_batch, Y_batch, M)

            predictions.append(pred)
            gradients.append(grad)

        # Verify shapes and types
        for i, (pred, grad) in enumerate(zip(predictions, gradients)):
            expected_pred_shape = (ntargets, ntasks, batch_sizes[i])
            expected_grad_shape = (nfeatures, ntargets, ntasks)

            validate_basic_properties(pred, expected_pred_shape, dtype_glm)
            validate_basic_properties(grad, expected_grad_shape, dtype_glm)

class TestSortIntegrationWorkflows:
    """Test sort operations in complex workflows."""

    def test_sort_then_statistical_operations(self, dtype_all):
        """Test sorting followed by statistical vector operations."""
        if not hasattr(py_gpu_algos, 'tensor_sort_bitonic'):
            pytest.skip("tensor_sort_bitonic not available")

        # Create 3D tensor with power-of-2 dimensions
        shape = (8, 4, 16)
        np.random.seed(42)

        if np.issubdtype(dtype_all, np.integer):
            if dtype_all in [np.int8, np.uint8]:
                tensor = np.random.randint(-20, 20, shape, dtype=dtype_all)
            else:
                tensor = np.random.randint(-100, 100, shape, dtype=dtype_all)
        else:
            tensor = (np.random.randn(*shape) * 50).astype(dtype_all)

        # Sort along different axes
        tensor_rows = tensor.copy()
        tensor_cols = tensor.copy()
        tensor_depth = tensor.copy()

        py_gpu_algos.tensor_sort_bitonic(tensor_rows, "rows")
        py_gpu_algos.tensor_sort_bitonic(tensor_cols, "cols")
        py_gpu_algos.tensor_sort_bitonic(tensor_depth, "sheets")

        # Extract vectors and apply vector operations
        # Take a slice and flatten for vector operations
        if dtype_all in [np.float16, np.float32, np.float64]:
            # Use floating point vector operations
            vec_from_sorted = tensor_depth[0, 0, :].copy()  # Extract a 1D slice

            cumsum_result = py_gpu_algos.vector_cumsum_parallel(vec_from_sorted)
            cummax_result = py_gpu_algos.vector_cummax_parallel(vec_from_sorted)

            # Verify that operations work on sorted data
            assert cumsum_result.shape == vec_from_sorted.shape
            assert cummax_result.shape == vec_from_sorted.shape

            # For sorted ascending data, cummax should equal the input
            assert_array_close(cummax_result, vec_from_sorted, dtype_all)

    def test_multi_axis_sorting_sequence(self, dtype_all):
        """Test sorting along multiple axes in sequence."""
        if not hasattr(py_gpu_algos, 'tensor_sort_bitonic'):
            pytest.skip("tensor_sort_bitonic not available")

        shape = (16, 8, 32)
        np.random.seed(42)

        if np.issubdtype(dtype_all, np.integer):
            tensor = np.random.randint(-50, 50, shape, dtype=dtype_all)
        else:
            tensor = (np.random.randn(*shape) * 30).astype(dtype_all)

        # Apply sorting sequence: rows -> cols -> sheets
        tensor_multi = tensor.copy()

        py_gpu_algos.tensor_sort_bitonic(tensor_multi, "rows")
        py_gpu_algos.tensor_sort_bitonic(tensor_multi, "cols")
        py_gpu_algos.tensor_sort_bitonic(tensor_multi, "sheets")

        # Final result should be sorted along sheets (last operation)
        for i in range(shape[0]):
            for j in range(shape[1]):
                sheets_slice = tensor_multi[i, j, :]
                sorted_slice = np.sort(sheets_slice)
                assert_array_close(sheets_slice, sorted_slice, dtype_all)

        # All original values should still be present
        original_flat = np.sort(tensor.flatten())
        result_flat = np.sort(tensor_multi.flatten())
        assert np.array_equal(original_flat, result_flat)

class TestCrossModuleDataFlow:
    """Test data flow and compatibility between different modules."""

    def test_dtype_preservation_across_operations(self, dtype_all):
        """Test that dtypes are preserved across different operations."""
        # Skip complex combinations for unsupported dtypes
        if dtype_all == np.complex64 or dtype_all == np.complex128:
            pytest.skip("Complex dtypes not supported")

        # Vector operations
        if dtype_all in [np.float16, np.float32, np.float64, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]:
            vec = np.random.randn(128).astype(dtype_all)
            if np.issubdtype(dtype_all, np.integer):
                vec = (vec * 10).astype(dtype_all)

            cumsum_result = py_gpu_algos.vector_cumsum_parallel(vec)
            assert cumsum_result.dtype == dtype_all

        # Matrix operations
        if dtype_all in [np.float16, np.float32, np.float64, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]:
            matrix = np.random.randn(16, 16).astype(dtype_all)
            if np.issubdtype(dtype_all, np.integer):
                matrix = (matrix * 10).astype(dtype_all)

            transpose_result = py_gpu_algos.matrix_transpose_striped(matrix)
            assert transpose_result.dtype == dtype_all

    def test_shape_compatibility_across_modules(self):
        """Test shape compatibility when passing data between modules."""
        dtype = np.float32

        # Create data that can flow between modules
        # Start with matrix operation
        a = np.random.randn(64, 48).astype(dtype)
        b = np.random.randn(48, 1).astype(dtype)

        # Matrix product gives column vector
        matrix_result = py_gpu_algos.matrix_product_naive(a, b)
        assert matrix_result.shape == (64, 1)

        # Convert to vector for vector operations
        vector_data = matrix_result.flatten()
        assert vector_data.shape == (64,)

        # Apply vector operations
        cumsum_result = py_gpu_algos.vector_cumsum_parallel(vector_data)
        assert cumsum_result.shape == (64,)

        # Reshape for further matrix operations
        if 64 == 8 * 8:  # Can reshape to square
            reshaped = cumsum_result.reshape(8, 8)
            transpose_result = py_gpu_algos.matrix_transpose_striped(reshaped)
            assert transpose_result.shape == (8, 8)

    def test_numerical_consistency_across_modules(self, dtype_float):
        """Test numerical consistency when combining operations."""
        np.random.seed(42)

        # Create test data
        matrix_a = np.random.randn(32, 24).astype(dtype_float)
        matrix_b = np.random.randn(24, 32).astype(dtype_float)

        # GPU computation: A @ B, then extract diagonal as vector
        product = py_gpu_algos.matrix_product_naive(matrix_a, matrix_b)
        diagonal = np.diag(product)  # Extract diagonal
        cumsum_diagonal = py_gpu_algos.vector_cumsum_parallel(diagonal)

        # NumPy reference computation
        numpy_product = get_numpy_reference_matrix_product(matrix_a, matrix_b)
        numpy_diagonal = np.diag(numpy_product)
        numpy_cumsum = get_numpy_reference_cumsum_parallel(numpy_diagonal)

        # Results should match
        assert_array_close(product, numpy_product, dtype_float)
        assert_array_close(cumsum_diagonal, numpy_cumsum, dtype_float)

class TestPerformanceIntegrationPatterns:
    """Test performance patterns in integrated workflows."""

    @pytest.mark.performance
    def test_matrix_chain_performance(self, performance_sizes):
        """Test performance of chained matrix operations."""
        dtype = np.float32

        for size, _, _ in performance_sizes:
            if size < 64:  # Skip small sizes for performance tests
                continue

            np.random.seed(42)
            a = np.random.randn(size, size).astype(dtype)
            b = np.random.randn(size, size).astype(dtype)

            # GPU chain: A @ B -> transpose -> @ A^T
            import time
            start_time = time.time()

            ab = py_gpu_algos.matrix_product_naive(a, b)
            ab_t = py_gpu_algos.matrix_transpose_striped(ab)
            a_t = py_gpu_algos.matrix_transpose_striped(a)
            result = py_gpu_algos.matrix_product_naive(ab_t, a_t)

            gpu_time = time.time() - start_time

            # NumPy reference
            start_time = time.time()
            numpy_ab = np.dot(a, b)
            numpy_ab_t = numpy_ab.T
            numpy_a_t = a.T
            numpy_result = np.dot(numpy_ab_t, numpy_a_t)
            numpy_time = time.time() - start_time

            # Verify correctness
            assert_array_close(result, numpy_result, dtype, rtol=1e-5)

            print(f"Matrix chain ({size}x{size}): GPU {gpu_time*1000:.2f}ms, NumPy {numpy_time*1000:.2f}ms")

    @pytest.mark.performance
    def test_glm_workflow_performance(self, performance_sizes):
        """Test performance of complete GLM workflow."""
        dtype = np.float32

        for base_size, _, _ in performance_sizes:
            if base_size < 64:
                continue

            # Scale GLM dimensions
            nfeatures = base_size
            ntargets = base_size // 8
            ntasks = 4
            nobs = base_size * 2

            np.random.seed(42)
            X = np.random.randn(nfeatures, ntasks, nobs).astype(dtype)
            Y = np.random.randn(ntargets, ntasks, nobs).astype(dtype)
            M = np.random.randn(nfeatures, ntargets, ntasks).astype(dtype)

            # GPU workflow: predict -> gradient
            import time
            start_time = time.time()

            pred = py_gpu_algos.glm_predict_naive(X, M)
            grad = py_gpu_algos.glm_gradient_naive(X, Y, M)

            gpu_time = time.time() - start_time

            # NumPy reference workflow
            start_time = time.time()
            numpy_pred = get_numpy_reference_glm_predict(X, M)
            numpy_grad = get_numpy_reference_glm_gradient(X, Y, M)
            numpy_time = time.time() - start_time

            # Verify correctness
            assert_array_close(pred, numpy_pred, dtype)
            assert_array_close(grad, numpy_grad, dtype)

            print(f"GLM workflow ({nfeatures}x{ntargets}x{ntasks}x{nobs}): GPU {gpu_time*1000:.2f}ms, NumPy {numpy_time*1000:.2f}ms")

class TestRealWorldUsagePatterns:
    """Test patterns that mimic real-world usage."""

    def test_machine_learning_pipeline(self, dtype_float):
        """Test a simplified machine learning pipeline."""
        np.random.seed(42)

        # Data preprocessing: normalize features (vector operations)
        features = np.random.randn(1000).astype(dtype_float) * 10 + 5
        normalized_features = features - np.mean(features)  # Center
        normalized_features = normalized_features / np.std(normalized_features)  # Scale

        # Compute running statistics with GPU
        cumsum_features = py_gpu_algos.vector_cumsum_parallel(normalized_features)
        cummax_features = py_gpu_algos.vector_cummax_parallel(np.abs(normalized_features))

        # Model training simulation (GLM operations)
        nfeatures, ntargets, ntasks, nobs = 20, 3, 2, 500
        X = np.random.randn(nfeatures, ntasks, nobs).astype(dtype_float)
        Y_true = np.random.randn(ntargets, ntasks, nobs).astype(dtype_float)
        M = np.random.randn(nfeatures, ntargets, ntasks).astype(dtype_float) * 0.1

        # Training steps
        for epoch in range(3):  # Simple training loop
            # Forward pass
            Y_pred = py_gpu_algos.glm_predict_naive(X, M)

            # Compute loss
            loss = np.mean((Y_pred - Y_true) ** 2)

            # Backward pass
            gradient = py_gpu_algos.glm_gradient_naive(X, Y_true, M)

            # Update parameters
            learning_rate = 0.01
            M = M - learning_rate * gradient

            # Verify shapes and loss decrease
            assert Y_pred.shape == Y_true.shape
            assert gradient.shape == M.shape
            if epoch > 0:
                assert loss <= prev_loss or abs(loss - prev_loss) < 1e-3  # Allow small fluctuations
            prev_loss = loss

    def test_data_analysis_workflow(self, dtype_float):
        """Test a data analysis workflow with sorting and statistics."""
        if not hasattr(py_gpu_algos, 'tensor_sort_bitonic'):
            pytest.skip("tensor_sort_bitonic not available")

        # Generate 3D dataset (experiments x conditions x measurements)
        shape = (8, 4, 32)  # Power-of-2 for sorting
        np.random.seed(42)
        dataset = (np.random.randn(*shape) * 10 + 50).astype(dtype_float)

        # Sort data along measurement axis for statistical analysis
        sorted_dataset = dataset.copy()
        py_gpu_algos.tensor_sort_bitonic(sorted_dataset, "sheets")

        # Extract statistics using vector operations
        # Take median values (middle elements after sorting)
        median_idx = shape[2] // 2
        medians = sorted_dataset[:, :, median_idx]  # Shape: (8, 4)

        # Compute cumulative statistics across experiments
        for condition in range(shape[1]):
            condition_medians = medians[:, condition]  # Vector of length 8

            # Cumulative sum and max across experiments
            cumsum_medians = py_gpu_algos.vector_cumsum_parallel(condition_medians)
            cummax_medians = py_gpu_algos.vector_cummax_parallel(condition_medians)

            # Verify results make sense
            assert cumsum_medians.shape == (shape[0],)
            assert cummax_medians.shape == (shape[0],)
            assert cumsum_medians[-1] == np.sum(condition_medians)  # Last element is total
            assert cummax_medians[-1] == np.max(condition_medians)  # Last element is max

    def test_numerical_computation_pipeline(self, dtype_float):
        """Test a numerical computation pipeline combining multiple operations."""
        np.random.seed(42)

        # Step 1: Matrix computations (simulate solving linear systems)
        A = np.random.randn(64, 64).astype(dtype_float) + np.eye(64, dtype=dtype_float) * 10  # Well-conditioned
        B = np.random.randn(64, 32).astype(dtype_float)

        # Compute A^T @ B using available operations
        A_T = py_gpu_algos.matrix_transpose_striped(A)
        result_matrix = py_gpu_algos.matrix_product_naive(A_T, B)

        # Step 2: Extract and analyze columns using vector operations
        for col in range(min(4, result_matrix.shape[1])):  # Analyze first few columns
            column_data = result_matrix[:, col].copy()

            # Compute running statistics
            cumsum_col = py_gpu_algos.vector_cumsum_parallel(column_data)
            cummax_col = py_gpu_algos.vector_cummax_parallel(np.abs(column_data))

            # Step 3: Use scan operations for further analysis
            scan_sum = py_gpu_algos.vector_scan_parallel(column_data, "sum")
            scan_max = py_gpu_algos.vector_scan_parallel(np.abs(column_data), "max")

            # Verify consistency
            assert_array_close(cumsum_col, scan_sum, dtype_float)
            assert_array_close(cummax_col, scan_max, dtype_float)

        # Verify overall computation with NumPy
        numpy_A_T = A.T
        numpy_result = np.dot(numpy_A_T, B)
        assert_array_close(result_matrix, numpy_result, dtype_float)
