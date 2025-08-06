"""
Tests for GLM (Generalized Linear Model) operations in py-gpu-algos.

Tests all 3 GLM kernels operating on 3D tensors for multitask learning:
- glm_predict_naive: Linear model prediction (X, M) -> Y_pred
- glm_gradient_naive: Gradient computation (X, Y, M) -> gradient
- glm_gradient_xyyhat: Optimized gradient computation (X, Y, M) -> gradient

These operations support restricted dtypes: float32, float64, int32, int64.
Tensor shapes: X(nfeatures, ntasks, nobs), Y(ntargets, ntasks, nobs), M(nfeatures, ntargets, ntasks)
"""

import pytest
import numpy as np
import py_gpu_algos
from .utils import (
    assert_array_close, validate_basic_properties, validate_function_error_cases,
    compare_with_numpy_reference, get_numpy_reference_glm_predict,
    get_numpy_reference_glm_gradient, ErrorCaseBuilder, print_performance_summary
)

# GLM function names to test
GLM_FUNCTIONS = [
    'glm_predict_naive',
    'glm_gradient_naive',
    'glm_gradient_xyyhat'
]

# GLM prediction functions (2in_1out pattern)
GLM_PREDICT_FUNCTIONS = ['glm_predict_naive']

# GLM gradient functions (3in_1out pattern)
GLM_GRADIENT_FUNCTIONS = ['glm_gradient_naive', 'glm_gradient_xyyhat']

class TestGLMPredict:
    """Test GLM prediction operations."""

    @pytest.mark.parametrize("func_name", GLM_PREDICT_FUNCTIONS)
    def test_basic_functionality(self, func_name, dtype_glm, glm_test_data):
        """Test basic GLM prediction functionality."""
        func = getattr(py_gpu_algos, func_name)
        X, Y, M = glm_test_data(dtype_glm)

        # Only need X and M for prediction
        result = func(X, M)

        # Validate basic properties
        expected_shape = Y.shape  # (ntargets, ntasks, nobs)
        validate_basic_properties(result, expected_shape, dtype_glm)

        # Compare with NumPy reference
        numpy_result = get_numpy_reference_glm_predict(X, M)
        assert_array_close(result, numpy_result, dtype_glm)

    @pytest.mark.parametrize("func_name", GLM_PREDICT_FUNCTIONS)
    def test_low_level_functions(self, func_name, dtype_glm, glm_test_data):
        """Test low-level type-specific functions."""
        dtype_name = dtype_glm.__name__
        low_level_func_name = f"{func_name}_{dtype_name}"

        # Check if low-level function exists
        if not hasattr(py_gpu_algos, low_level_func_name):
            pytest.skip(f"Low-level function {low_level_func_name} not available")

        low_level_func = getattr(py_gpu_algos, low_level_func_name)
        high_level_func = getattr(py_gpu_algos, func_name)

        X, Y, M = glm_test_data(dtype_glm)

        # Compare low-level and high-level results
        result_low = low_level_func(X, M)
        result_high = high_level_func(X, M)

        assert_array_close(result_low, result_high, dtype_glm)

    @pytest.mark.parametrize("func_name", GLM_PREDICT_FUNCTIONS)
    def test_different_shapes(self, func_name, dtype_glm):
        """Test GLM prediction with different tensor shapes."""
        func = getattr(py_gpu_algos, func_name)

        test_shapes = [
            (5, 3, 2, 50),    # Small tensors
            (20, 8, 4, 100),  # Medium tensors
            (10, 1, 5, 200),  # Single target
            (15, 5, 1, 100),  # Single task
        ]

        for nfeatures, ntargets, ntasks, nobs in test_shapes:
            np.random.seed(42)
            X = np.random.randn(nfeatures, ntasks, nobs).astype(dtype_glm)
            M = np.random.randn(nfeatures, ntargets, ntasks).astype(dtype_glm)

            # Scale for integer types
            if np.issubdtype(dtype_glm, np.integer):
                scale = 10
                X = (X * scale).astype(dtype_glm)
                M = (M * scale).astype(dtype_glm)

            result = func(X, M)
            numpy_result = get_numpy_reference_glm_predict(X, M)

            expected_shape = (ntargets, ntasks, nobs)
            validate_basic_properties(result, expected_shape, dtype_glm)
            assert_array_close(result, numpy_result, dtype_glm)

    @pytest.mark.parametrize("func_name", GLM_PREDICT_FUNCTIONS)
    def test_mathematical_properties(self, func_name, dtype_glm):
        """Test mathematical properties of GLM prediction."""
        func = getattr(py_gpu_algos, func_name)

        nfeatures, ntargets, ntasks, nobs = 10, 3, 4, 100
        np.random.seed(42)

        X = np.random.randn(nfeatures, ntasks, nobs).astype(dtype_glm)
        M = np.random.randn(nfeatures, ntargets, ntasks).astype(dtype_glm)

        if np.issubdtype(dtype_glm, np.integer):
            scale = 5
            X = (X * scale).astype(dtype_glm)
            M = (M * scale).astype(dtype_glm)

        # Test linearity: prediction with 2*M should give 2*prediction
        result1 = func(X, M)
        result2 = func(X, 2 * M)

        if dtype_glm in [np.float32, np.float64]:
            # For floating point, should be exactly 2x
            assert_array_close(result2, 2 * result1, dtype_glm)

        # Test zero model gives zero prediction
        M_zero = np.zeros_like(M)
        result_zero = func(X, M_zero)
        expected_zero = np.zeros((ntargets, ntasks, nobs), dtype=dtype_glm)
        assert_array_close(result_zero, expected_zero, dtype_glm)

    @pytest.mark.parametrize("func_name", GLM_PREDICT_FUNCTIONS)
    def test_error_handling(self, func_name, dtype_glm):
        """Test error handling for invalid inputs."""
        func = getattr(py_gpu_algos, func_name)

        # Create test tensors
        X_good = np.random.randn(10, 4, 100).astype(dtype_glm)
        M_good = np.random.randn(10, 3, 4).astype(dtype_glm)

        # Wrong shapes
        X_wrong_features = np.random.randn(8, 4, 100).astype(dtype_glm)  # 8 != 10
        M_wrong_tasks = np.random.randn(10, 3, 5).astype(dtype_glm)      # 5 != 4

        # Wrong dimensions
        X_2d = np.random.randn(10, 100).astype(dtype_glm)
        M_2d = np.random.randn(10, 3).astype(dtype_glm)
        X_4d = np.random.randn(10, 4, 100, 5).astype(dtype_glm)

        # Different dtypes
        X_float32 = X_good.astype(np.float32)
        M_float64 = M_good.astype(np.float64)

        error_cases = (ErrorCaseBuilder()
            .add_dimension_mismatch(func_name, X_wrong_features, M_good)
            .add_dimension_mismatch(func_name, X_good, M_wrong_tasks)
            .add_shape_error(func_name, X_2d, M_good)
            .add_shape_error(func_name, X_good, M_2d)
            .add_shape_error(func_name, X_4d, M_good)
            .add_dtype_mismatch(func_name, X_float32, M_float64)
            .build())

        validate_function_error_cases(func, error_cases)

class TestGLMGradient:
    """Test GLM gradient operations."""

    @pytest.mark.parametrize("func_name", GLM_GRADIENT_FUNCTIONS)
    def test_basic_functionality(self, func_name, dtype_glm, glm_test_data):
        """Test basic GLM gradient functionality."""
        func = getattr(py_gpu_algos, func_name)
        X, Y, M = glm_test_data(dtype_glm)

        # Compute gradient
        result = func(X, Y, M)

        # Validate basic properties
        expected_shape = M.shape  # (nfeatures, ntargets, ntasks)
        validate_basic_properties(result, expected_shape, dtype_glm)

        # Compare with NumPy reference
        numpy_result = get_numpy_reference_glm_gradient(X, Y, M)
        assert_array_close(result, numpy_result, dtype_glm)

    @pytest.mark.parametrize("func_name", GLM_GRADIENT_FUNCTIONS)
    def test_low_level_functions(self, func_name, dtype_glm, glm_test_data):
        """Test low-level type-specific functions."""
        dtype_name = dtype_glm.__name__
        low_level_func_name = f"{func_name}_{dtype_name}"

        # Check if low-level function exists
        if not hasattr(py_gpu_algos, low_level_func_name):
            pytest.skip(f"Low-level function {low_level_func_name} not available")

        low_level_func = getattr(py_gpu_algos, low_level_func_name)
        high_level_func = getattr(py_gpu_algos, func_name)

        X, Y, M = glm_test_data(dtype_glm)

        # Compare low-level and high-level results
        result_low = low_level_func(X, Y, M)
        result_high = high_level_func(X, Y, M)

        assert_array_close(result_low, result_high, dtype_glm)

    def test_gradient_algorithms_consistency(self, dtype_glm, glm_test_data):
        """Test that different gradient algorithms give consistent results."""
        X, Y, M = glm_test_data(dtype_glm)

        # Run both gradient algorithms
        gradient_naive = py_gpu_algos.glm_gradient_naive(X, Y, M)
        gradient_xyyhat = py_gpu_algos.glm_gradient_xyyhat(X, Y, M)

        # Results should be identical (or very close for floating point)
        if dtype_glm in [np.float32, np.float64]:
            # Allow small numerical differences between algorithms
            rtol = 1e-4 if dtype_glm == np.float32 else 1e-10
            assert_array_close(gradient_naive, gradient_xyyhat, dtype_glm, rtol=rtol)
        else:
            # Integer arithmetic should be exact
            assert_array_close(gradient_naive, gradient_xyyhat, dtype_glm)

    @pytest.mark.parametrize("func_name", GLM_GRADIENT_FUNCTIONS)
    def test_different_shapes(self, func_name, dtype_glm):
        """Test GLM gradient with different tensor shapes."""
        func = getattr(py_gpu_algos, func_name)

        test_shapes = [
            (5, 3, 2, 50),    # Small tensors
            (20, 8, 4, 100),  # Medium tensors
            (10, 1, 5, 200),  # Single target
            (15, 5, 1, 100),  # Single task
        ]

        for nfeatures, ntargets, ntasks, nobs in test_shapes:
            np.random.seed(42)
            X = np.random.randn(nfeatures, ntasks, nobs).astype(dtype_glm)
            Y = np.random.randn(ntargets, ntasks, nobs).astype(dtype_glm)
            M = np.random.randn(nfeatures, ntargets, ntasks).astype(dtype_glm)

            # Scale for integer types
            if np.issubdtype(dtype_glm, np.integer):
                scale = 5
                X = (X * scale).astype(dtype_glm)
                Y = (Y * scale).astype(dtype_glm)
                M = (M * scale).astype(dtype_glm)

            result = func(X, Y, M)
            numpy_result = get_numpy_reference_glm_gradient(X, Y, M)

            expected_shape = (nfeatures, ntargets, ntasks)
            validate_basic_properties(result, expected_shape, dtype_glm)
            assert_array_close(result, numpy_result, dtype_glm)

    @pytest.mark.parametrize("func_name", GLM_GRADIENT_FUNCTIONS)
    def test_mathematical_properties(self, func_name, dtype_glm):
        """Test mathematical properties of gradient computation."""
        func = getattr(py_gpu_algos, func_name)

        nfeatures, ntargets, ntasks, nobs = 8, 3, 4, 100
        np.random.seed(42)

        X = np.random.randn(nfeatures, ntasks, nobs).astype(dtype_glm)
        Y = np.random.randn(ntargets, ntasks, nobs).astype(dtype_glm)
        M = np.random.randn(nfeatures, ntargets, ntasks).astype(dtype_glm)

        if np.issubdtype(dtype_glm, np.integer):
            scale = 3
            X = (X * scale).astype(dtype_glm)
            Y = (Y * scale).astype(dtype_glm)
            M = (M * scale).astype(dtype_glm)

        # Test that gradient is zero when prediction matches target
        Y_pred = py_gpu_algos.glm_predict_naive(X, M)
        gradient_zero = func(X, Y_pred, M)

        # Gradient should be close to zero (within numerical precision)
        if dtype_glm in [np.float32, np.float64]:
            rtol = 1e-5 if dtype_glm == np.float32 else 1e-12
            expected_zero = np.zeros_like(M)
            assert_array_close(gradient_zero, expected_zero, dtype_glm, rtol=rtol, atol=1e-6)

    @pytest.mark.parametrize("func_name", GLM_GRADIENT_FUNCTIONS)
    def test_error_handling(self, func_name, dtype_glm):
        """Test error handling for invalid inputs."""
        func = getattr(py_gpu_algos, func_name)

        # Create test tensors
        X_good = np.random.randn(10, 4, 100).astype(dtype_glm)
        Y_good = np.random.randn(3, 4, 100).astype(dtype_glm)
        M_good = np.random.randn(10, 3, 4).astype(dtype_glm)

        # Wrong shapes
        X_wrong_features = np.random.randn(8, 4, 100).astype(dtype_glm)
        Y_wrong_targets = np.random.randn(5, 4, 100).astype(dtype_glm)
        Y_wrong_tasks = np.random.randn(3, 5, 100).astype(dtype_glm)
        Y_wrong_obs = np.random.randn(3, 4, 80).astype(dtype_glm)
        M_wrong_features = np.random.randn(8, 3, 4).astype(dtype_glm)
        M_wrong_targets = np.random.randn(10, 5, 4).astype(dtype_glm)
        M_wrong_tasks = np.random.randn(10, 3, 5).astype(dtype_glm)

        # Wrong dimensions
        X_2d = np.random.randn(10, 100).astype(dtype_glm)
        Y_2d = np.random.randn(3, 100).astype(dtype_glm)

        # Different dtypes
        X_float32 = X_good.astype(np.float32)
        Y_float64 = Y_good.astype(np.float64)

        error_cases = (ErrorCaseBuilder()
            .add_dimension_mismatch(func_name, X_wrong_features, Y_good, M_good)
            .add_dimension_mismatch(func_name, X_good, Y_wrong_targets, M_good)
            .add_dimension_mismatch(func_name, X_good, Y_wrong_tasks, M_good)
            .add_dimension_mismatch(func_name, X_good, Y_wrong_obs, M_good)
            .add_dimension_mismatch(func_name, X_good, Y_good, M_wrong_features)
            .add_dimension_mismatch(func_name, X_good, Y_good, M_wrong_targets)
            .add_dimension_mismatch(func_name, X_good, Y_good, M_wrong_tasks)
            .add_shape_error(func_name, X_2d, Y_good, M_good)
            .add_shape_error(func_name, X_good, Y_2d, M_good)
            .add_dtype_mismatch(func_name, X_float32, Y_float64, M_good)
            .build())

        validate_function_error_cases(func, error_cases)

class TestGLMOperationsIntegration:
    """Integration tests combining GLM operations."""

    def test_predict_then_gradient_workflow(self, dtype_glm, glm_test_data):
        """Test typical ML workflow: predict then compute gradient."""
        X, Y_true, M = glm_test_data(dtype_glm)

        # Forward pass: prediction
        Y_pred = py_gpu_algos.glm_predict_naive(X, M)

        # Backward pass: gradient computation using true targets
        gradient = py_gpu_algos.glm_gradient_naive(X, Y_true, M)

        # Validate shapes
        assert Y_pred.shape == Y_true.shape
        assert gradient.shape == M.shape

        # Compare with NumPy references
        numpy_pred = get_numpy_reference_glm_predict(X, M)
        numpy_grad = get_numpy_reference_glm_gradient(X, Y_true, M)

        assert_array_close(Y_pred, numpy_pred, dtype_glm)
        assert_array_close(gradient, numpy_grad, dtype_glm)

    def test_gradient_consistency_across_algorithms(self, dtype_glm):
        """Test that both gradient algorithms produce consistent results."""
        np.random.seed(42)
        nfeatures, ntargets, ntasks, nobs = 12, 4, 3, 150

        X = np.random.randn(nfeatures, ntasks, nobs).astype(dtype_glm)
        Y = np.random.randn(ntargets, ntasks, nobs).astype(dtype_glm)
        M = np.random.randn(nfeatures, ntargets, ntasks).astype(dtype_glm)

        if np.issubdtype(dtype_glm, np.integer):
            scale = 5
            X = (X * scale).astype(dtype_glm)
            Y = (Y * scale).astype(dtype_glm)
            M = (M * scale).astype(dtype_glm)

        # Both gradient algorithms should give same result
        grad_naive = py_gpu_algos.glm_gradient_naive(X, Y, M)
        grad_xyyhat = py_gpu_algos.glm_gradient_xyyhat(X, Y, M)

        # Allow small numerical differences for floating point
        if dtype_glm in [np.float32, np.float64]:
            rtol = 1e-4 if dtype_glm == np.float32 else 1e-10
            assert_array_close(grad_naive, grad_xyyhat, dtype_glm, rtol=rtol)
        else:
            assert_array_close(grad_naive, grad_xyyhat, dtype_glm)

    def test_optimization_step_simulation(self, dtype_glm):
        """Simulate a gradient descent optimization step."""
        if dtype_glm not in [np.float32, np.float64]:
            pytest.skip("Optimization simulation only meaningful for floating point")

        np.random.seed(42)
        nfeatures, ntargets, ntasks, nobs = 8, 2, 3, 200

        # Generate synthetic data
        X = np.random.randn(nfeatures, ntasks, nobs).astype(dtype_glm)
        M_true = np.random.randn(nfeatures, ntargets, ntasks).astype(dtype_glm) * 0.1

        # Generate true targets using true model
        Y_true = py_gpu_algos.glm_predict_naive(X, M_true)

        # Initialize model with random weights
        M_init = np.random.randn(nfeatures, ntargets, ntasks).astype(dtype_glm) * 0.1

        # Compute initial loss (mean squared error)
        Y_pred_init = py_gpu_algos.glm_predict_naive(X, M_init)
        loss_init = np.mean((Y_pred_init - Y_true) ** 2)

        # Compute gradient
        gradient = py_gpu_algos.glm_gradient_naive(X, Y_true, M_init)

        # Take a small gradient step
        learning_rate = 0.001
        M_updated = M_init - learning_rate * gradient

        # Compute new loss
        Y_pred_updated = py_gpu_algos.glm_predict_naive(X, M_updated)
        loss_updated = np.mean((Y_pred_updated - Y_true) ** 2)

        # Loss should decrease (gradient descent property)
        assert loss_updated < loss_init, f"Loss should decrease: {loss_init} -> {loss_updated}"

    def test_prediction_gradient_mathematical_relationship(self, dtype_glm):
        """Test mathematical relationship between prediction and gradient."""
        if dtype_glm not in [np.float32, np.float64]:
            pytest.skip("Mathematical relationship test only for floating point")

        np.random.seed(42)
        nfeatures, ntargets, ntasks, nobs = 6, 2, 2, 100

        X = np.random.randn(nfeatures, ntasks, nobs).astype(dtype_glm)
        Y = np.random.randn(ntargets, ntasks, nobs).astype(dtype_glm)
        M = np.random.randn(nfeatures, ntargets, ntasks).astype(dtype_glm)

        # Test: gradient should be X @ (Y - Y_pred)^T for each task
        Y_pred = py_gpu_algos.glm_predict_naive(X, M)
        gradient = py_gpu_algos.glm_gradient_naive(X, Y, M)

        # Manually compute gradient using NumPy
        manual_gradient = np.zeros_like(M)
        for task in range(ntasks):
            residual = Y[:, task, :] - Y_pred[:, task, :]  # (ntargets, nobs)
            manual_gradient[:, :, task] = X[:, task, :] @ residual.T  # (nfeatures, nobs) @ (nobs, ntargets)

        assert_array_close(gradient, manual_gradient, dtype_glm, rtol=1e-5)

class TestGLMOperationsPerformance:
    """Performance tests for GLM operations."""

    @pytest.mark.performance
    @pytest.mark.parametrize("func_name", GLM_PREDICT_FUNCTIONS)
    def test_predict_performance(self, func_name, performance_sizes):
        """Test GLM prediction performance."""
        func = getattr(py_gpu_algos, func_name)
        dtype = np.float32

        for base_size, _, _ in performance_sizes:
            # Scale dimensions appropriately for GLM
            nfeatures = base_size
            ntargets = base_size // 8
            ntasks = 4
            nobs = base_size * 4

            np.random.seed(42)
            X = np.random.randn(nfeatures, ntasks, nobs).astype(dtype)
            M = np.random.randn(nfeatures, ntargets, ntasks).astype(dtype)

            results = compare_with_numpy_reference(
                func, get_numpy_reference_glm_predict,
                (X, M), dtype
            )

            assert results['accuracy_pass'], f"Accuracy failed for {func_name}"

            if nfeatures >= 128:
                print_performance_summary(results,
                    f"{func_name} ({nfeatures}x{ntargets}x{ntasks}x{nobs})")

    @pytest.mark.performance
    @pytest.mark.parametrize("func_name", GLM_GRADIENT_FUNCTIONS)
    def test_gradient_performance(self, func_name, performance_sizes):
        """Test GLM gradient performance."""
        func = getattr(py_gpu_algos, func_name)
        dtype = np.float32

        for base_size, _, _ in performance_sizes:
            # Scale dimensions for gradient computation
            nfeatures = base_size
            ntargets = base_size // 8
            ntasks = 4
            nobs = base_size * 2

            np.random.seed(42)
            X = np.random.randn(nfeatures, ntasks, nobs).astype(dtype)
            Y = np.random.randn(ntargets, ntasks, nobs).astype(dtype)
            M = np.random.randn(nfeatures, ntargets, ntasks).astype(dtype)

            results = compare_with_numpy_reference(
                func, get_numpy_reference_glm_gradient,
                (X, Y, M), dtype
            )

            assert results['accuracy_pass'], f"Accuracy failed for {func_name}"

            if nfeatures >= 128:
                print_performance_summary(results,
                    f"{func_name} ({nfeatures}x{ntargets}x{ntasks}x{nobs})")

class TestGLMOperationsEdgeCases:
    """Test edge cases for GLM operations."""

    def test_single_dimensions(self, dtype_glm):
        """Test GLM operations with single-element dimensions."""
        # Single feature
        X_single_feat = np.random.randn(1, 3, 100).astype(dtype_glm)
        M_single_feat = np.random.randn(1, 2, 3).astype(dtype_glm)
        Y_single_target = np.random.randn(1, 3, 100).astype(dtype_glm)

        # Single task
        X_single_task = np.random.randn(5, 1, 100).astype(dtype_glm)
        M_single_task = np.random.randn(5, 2, 1).astype(dtype_glm)
        Y_single_task = np.random.randn(2, 1, 100).astype(dtype_glm)

        # Single observation
        X_single_obs = np.random.randn(5, 3, 1).astype(dtype_glm)
        M_single_obs = np.random.randn(5, 2, 3).astype(dtype_glm)
        Y_single_obs = np.random.randn(2, 3, 1).astype(dtype_glm)

        if np.issubdtype(dtype_glm, np.integer):
            scale = 10
            X_single_feat = (X_single_feat * scale).astype(dtype_glm)
            M_single_feat = (M_single_feat * scale).astype(dtype_glm)
            Y_single_target = (Y_single_target * scale).astype(dtype_glm)
            X_single_task = (X_single_task * scale).astype(dtype_glm)
            M_single_task = (M_single_task * scale).astype(dtype_glm)
            Y_single_task = (Y_single_task * scale).astype(dtype_glm)
            X_single_obs = (X_single_obs * scale).astype(dtype_glm)
            M_single_obs = (M_single_obs * scale).astype(dtype_glm)
            Y_single_obs = (Y_single_obs * scale).astype(dtype_glm)

        # Test all operations with different single dimensions
        test_cases = [
            (X_single_feat, M_single_feat, Y_single_target),
            (X_single_task, M_single_task, Y_single_task),
            (X_single_obs, M_single_obs, Y_single_obs),
        ]

        for X, M, Y in test_cases:
            # Test prediction
            pred_result = py_gpu_algos.glm_predict_naive(X, M)
            numpy_pred = get_numpy_reference_glm_predict(X, M)
            assert_array_close(pred_result, numpy_pred, dtype_glm)

            # Test gradient
            grad_result = py_gpu_algos.glm_gradient_naive(X, Y, M)
            numpy_grad = get_numpy_reference_glm_gradient(X, Y, M)
            assert_array_close(grad_result, numpy_grad, dtype_glm)

    def test_zero_tensors(self, dtype_glm):
        """Test GLM operations with zero tensors."""
        nfeatures, ntargets, ntasks, nobs = 5, 3, 2, 50

        X = np.random.randn(nfeatures, ntasks, nobs).astype(dtype_glm)
        Y = np.random.randn(ntargets, ntasks, nobs).astype(dtype_glm)
        M = np.random.randn(nfeatures, ntargets, ntasks).astype(dtype_glm)

        if np.issubdtype(dtype_glm, np.integer):
            scale = 5
            X = (X * scale).astype(dtype_glm)
            Y = (Y * scale).astype(dtype_glm)
            M = (M * scale).astype(dtype_glm)

        # Zero input features
        X_zero = np.zeros_like(X)
        pred_zero_X = py_gpu_algos.glm_predict_naive(X_zero, M)
        expected_zero_pred = np.zeros((ntargets, ntasks, nobs), dtype=dtype_glm)
        assert_array_close(pred_zero_X, expected_zero_pred, dtype_glm)

        # Zero model
        M_zero = np.zeros_like(M)
        pred_zero_M = py_gpu_algos.glm_predict_naive(X, M_zero)
        assert_array_close(pred_zero_M, expected_zero_pred, dtype_glm)

        # Zero targets (gradient should be negative of prediction scaled by X)
        Y_zero = np.zeros_like(Y)
        grad_zero_Y = py_gpu_algos.glm_gradient_naive(X, Y_zero, M)
        numpy_grad_zero_Y = get_numpy_reference_glm_gradient(X, Y_zero, M)
        assert_array_close(grad_zero_Y, numpy_grad_zero_Y, dtype_glm)
