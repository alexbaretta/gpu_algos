#!/usr/bin/env python3

import sys
import ijson
from pathlib import Path
from collections import defaultdict

def read_streaming_results(json_file):
    """Read streaming JSON results using json_stream for robust parsing."""
    results = defaultdict(list)

    with open(json_file, 'r') as f:
        try:
            # json_stream.load() can handle both JSON Lines and other streaming formats
            for result in ijson.items(f, '', multiple_values=True):
                # Group results by executable name
                if 'executable' in result:
                    results[result['executable']].append(result)
                else:
                    # Handle error records that might not have executable field
                    results['_errors'].append(result)
        except Exception as e:
            print(f"Error parsing JSON stream: {e}", file=sys.stderr)
            return {}

    return dict(results)

def analyze_results(json_file):
    results = read_streaming_results(json_file)

    print("=" * 80)
    print("GPU ALGORITHM TEST SUMMARY REPORT")
    print("=" * 80)
    print()

    # Handle errors separately
    errors = results.pop('_errors', [])

    total_executables = len(results)
    total_tests = 0
    total_passed = 0
    total_correct = 0

    executable_summary = []

    for exe_name, exe_results in results.items():
        exe_total = len(exe_results)
        exe_passed = sum(1 for r in exe_results if r.get("run_success", False))
        exe_correct = sum(1 for r in exe_results if r.get("correct", False))

        total_tests += exe_total
        total_passed += exe_passed
        total_correct += exe_correct

        status = "PASS" if exe_passed == exe_total else "FAIL"
        correctness = f"{exe_correct}/{exe_total}"

        executable_summary.append({
            'name': exe_name,
            'status': status,
            'execution': f"{exe_passed}/{exe_total}",
            'correctness': correctness,
            'exe_passed': exe_passed,
            'exe_total': exe_total,
            'exe_correct': exe_correct
        })

    # Sort by name for consistent output
    executable_summary.sort(key=lambda x: x['name'])

    print(f"{'Executable':<40} {'Status':<8} {'Execution':<12} {'Correctness':<12}")
    print("-" * 80)

    for exe in executable_summary:
        print(f"{exe['name']:<40} {exe['status']:<8} {exe['execution']:<12} {exe['correctness']:<12}")

    print()
    print("=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)
    print(f"Total Executables:     {total_executables}")
    print(f"Total Tests:           {total_tests}")
    print(f"Successful Executions: {total_passed}/{total_tests} ({total_passed/total_tests*100:.1f}%)")
    print(f"Correct Results:       {total_correct}/{total_tests} ({total_correct/total_tests*100:.1f}%)")

    if errors:
        print(f"Parse/General Errors:  {len(errors)}")

    print()

    # Categorize by algorithm type
    categories = {
        'gradient': [exe for exe in executable_summary if exe['name'].startswith('gradient_')],
        'matrix_product': [exe for exe in executable_summary if exe['name'].startswith('matrix_product_')],
        'matrix_transpose': [exe for exe in executable_summary if exe['name'].startswith('matrix_transpose_')],
        'vector': [exe for exe in executable_summary if exe['name'].startswith('vector_')],
        'tensor': [exe for exe in executable_summary if exe['name'].startswith('tensor_')],
    }

    print("ALGORITHM CATEGORY BREAKDOWN")
    print("=" * 80)
    for category, exes in categories.items():
        if exes:
            cat_passed = sum(exe['exe_passed'] for exe in exes)
            cat_total = sum(exe['exe_total'] for exe in exes)
            cat_correct = sum(exe['exe_correct'] for exe in exes)

            print(f"{category.upper()}:")
            print(f"  Executables: {len(exes)}")
            print(f"  Execution:   {cat_passed}/{cat_total} ({cat_passed/cat_total*100:.1f}%)")
            print(f"  Correctness: {cat_correct}/{cat_total} ({cat_correct/cat_total*100:.1f}%)")
            print()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 analyze_results.py <json_file>")
        sys.exit(1)

    analyze_results(sys.argv[1])
