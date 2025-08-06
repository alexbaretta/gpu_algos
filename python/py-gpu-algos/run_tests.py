#!/usr/bin/env python3
"""
Test runner script for py-gpu-algos test suite.

Provides convenient commands for running different test categories and configurations.
"""

import argparse
import sys
import os
import subprocess
import time

def run_command(cmd, description=""):
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    if description:
        print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)

    start_time = time.time()
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    end_time = time.time()

    print(f"\nCompleted in {end_time - start_time:.2f} seconds")
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(description="Run py-gpu-algos tests")

    # Test selection options
    parser.add_argument('--fast', action='store_true',
                       help='Run only fast tests (exclude slow and performance tests)')
    parser.add_argument('--slow', action='store_true',
                       help='Run only slow tests')
    parser.add_argument('--performance', action='store_true',
                       help='Run only performance tests')
    parser.add_argument('--integration', action='store_true',
                       help='Run only integration tests')
    parser.add_argument('--error-handling', action='store_true',
                       help='Run only error handling tests')

    # Module-specific options
    parser.add_argument('--matrix', action='store_true',
                       help='Run only matrix operations tests')
    parser.add_argument('--vector', action='store_true',
                       help='Run only vector operations tests')
    parser.add_argument('--glm', action='store_true',
                       help='Run only GLM operations tests')
    parser.add_argument('--sort', action='store_true',
                       help='Run only sort operations tests')

    # Coverage and output options
    parser.add_argument('--coverage', action='store_true',
                       help='Run tests with coverage reporting')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Quiet output')
    parser.add_argument('--parallel', '-n', type=int, metavar='N',
                       help='Run tests in parallel with N workers')

    # Output options
    parser.add_argument('--html-report', action='store_true',
                       help='Generate HTML test report')
    parser.add_argument('--junit-xml', metavar='FILE',
                       help='Generate JUnit XML report')

    # Specific test patterns
    parser.add_argument('pattern', nargs='?',
                       help='Run tests matching pattern (e.g., test_matrix_ops.py::TestMatrixProducts)')

    args = parser.parse_args()

    # Build pytest command
    cmd = ['python', '-m', 'pytest']

    # Add test selection markers
    markers = []
    if args.fast:
        markers.append('not slow and not performance')
    elif args.slow:
        markers.append('slow')
    elif args.performance:
        markers.append('performance')
    elif args.integration:
        markers.append('integration')
    elif args.error_handling:
        markers.append('error_handling')

    # Add module-specific markers
    module_markers = []
    if args.matrix:
        module_markers.append('matrix_ops')
    if args.vector:
        module_markers.append('vector_ops')
    if args.glm:
        module_markers.append('glm_ops')
    if args.sort:
        module_markers.append('sort_ops')

    if module_markers:
        markers.append(' or '.join(module_markers))

    if markers:
        cmd.extend(['-m', ' and '.join(f'({m})' for m in markers)])

    # Add verbosity options
    if args.verbose:
        cmd.append('-v')
    elif args.quiet:
        cmd.append('-q')

    # Add parallel execution
    if args.parallel:
        cmd.extend(['-n', str(args.parallel)])

    # Add coverage
    if args.coverage:
        cmd.extend(['--cov=py_gpu_algos', '--cov-report=term-missing', '--cov-report=html'])

    # Add HTML report
    if args.html_report:
        cmd.extend(['--html=test_report.html', '--self-contained-html'])

    # Add JUnit XML
    if args.junit_xml:
        cmd.extend(['--junit-xml', args.junit_xml])

    # Add specific pattern
    if args.pattern:
        cmd.append(args.pattern)
    else:
        cmd.append('tests/')

    # Print test configuration summary
    print("py-gpu-algos Test Runner")
    print("=" * 60)
    print(f"Test selection: {' '.join(markers) if markers else 'All tests'}")
    if args.coverage:
        print("Coverage: Enabled")
    if args.parallel:
        print(f"Parallel workers: {args.parallel}")

    # Check if CUDA is available
    try:
        import py_gpu_algos
        cuda_available = py_gpu_algos._CUDA_AVAILABLE if hasattr(py_gpu_algos, '_CUDA_AVAILABLE') else True
        print(f"CUDA availability: {'Available' if cuda_available else 'Not available'}")
    except ImportError:
        print("CUDA availability: py_gpu_algos not installed")

    # Run the tests
    success = run_command(cmd, "pytest test suite")

    if success:
        print("\n✅ All tests passed!")

        # Show quick stats if not quiet
        if not args.quiet:
            print("\nQuick test commands:")
            print("  python run_tests.py --fast          # Fast tests only")
            print("  python run_tests.py --performance   # Performance tests")
            print("  python run_tests.py --matrix        # Matrix operations only")
            print("  python run_tests.py --coverage      # With coverage report")
    else:
        print("\n❌ Some tests failed!")
        return 1

    return 0

if __name__ == '__main__':
    sys.exit(main())
