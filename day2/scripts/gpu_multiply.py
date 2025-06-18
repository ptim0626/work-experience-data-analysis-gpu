#!/usr/bin/env python3
"""
GPU Multiply - Operation other than sum
============================================
This programme demonstrates a different GPU operation.
Instead of adding, we multiply every element by 2.
"""

import numpy as np
import cupy as cp
from pathlib import Path
import sys

def main():
    print("="*60)
    print("GPU MULTIPLY - DOUBLING 1000 NUMBERS")
    print("="*60)

    # ========================================
    # STEP 1: Load the GPU kernel from file
    # ========================================
    print("\n[STEP 1] Loading CUDA kernel from file...")
    kernel_file = Path("kernel_multiply.cu")

    # check if kernel file exists
    if not kernel_file.exists():
        raise FileNotFoundError(
            f"Kernel file '{kernel_file}' not found! "
            "Make sure the .cu file is in the same directory as this script."
        )

    # read the kernel code from file
    kernel_code = kernel_file.read_text()
    print(f"Successfully loaded kernel from {kernel_file}")

    # compile the kernel
    multiply_kernel = cp.RawKernel(kernel_code, 'multiply')
    print("Kernel compiled successfully")

    # ==========================================
    # STEP 2: Create test data on the CPU
    # ==========================================
    print("\n[STEP 2] Creating test data...")
    array_size = 1000
    cpu_data = np.arange(array_size, dtype=np.float32)

    print(f"Created array with {array_size} elements")
    print(f"First 5 elements: {cpu_data[:5]}")
    print(f"Last 5 elements:  {cpu_data[-5:]}")

    # =============================================
    # STEP 3: Copy data from CPU to GPU memory
    # =============================================
    print("\n[STEP 3] Transferring data to GPU...")
    gpu_data = cp.asarray(cpu_data)

    print("Data copied to GPU memory")

    # ===============================================
    # STEP 4: Configure the GPU execution layout
    # ===============================================
    print("\n[STEP 4] Configuring GPU execution...")

    threads_per_block = 256
    blocks_per_grid = (array_size + threads_per_block - 1) // threads_per_block

    total_threads = blocks_per_grid * threads_per_block
    print(f"GPU configuration:")
    print(f"Array size:        {array_size} elements")
    print(f"Threads per block: {threads_per_block}")
    print(f"Number of blocks:  {blocks_per_grid}")
    print(f"Total threads:     {total_threads}")

    # ===================================
    # STEP 5: Launch the kernel on GPU
    # ===================================
    print("\n[STEP 5] Launching GPU kernel...")
    print("All threads multiplying simultaneously.")

    multiply_kernel(
        (blocks_per_grid,),
        (threads_per_block,),
        (gpu_data, array_size)
    )

    print("GPU computation completed")

    # =============================================
    # STEP 6: Copy results back and verify
    # =============================================
    print("\n[STEP 6] Retrieving results...")
    result = cp.asnumpy(gpu_data)

    print("Results copied back to CPU")
    print(f"First 5 elements: {result[:5]}")
    print(f"Last 5 elements:  {result[-5:]}")

    # verify correctness
    print("\n[VERIFICATION] Checking results...")
    expected = cpu_data * 2
    errors = np.abs(result - expected)
    max_error = np.max(errors)

    if np.allclose(result, expected):
        print("SUCCESS! All values correctly doubled")
    else:
        print("ERROR: Results don't match expected values")
    print(f"Maximum error: {max_error}")

    # demonstrate the transformation
    print("\n[TRANSFORMATION EXAMPLE]")
    print("Original -> Result")
    for i in range(5):
        print(f"{cpu_data[i]:8.1f} -> {result[i]:8.1f}")

    # ========================================
    # SUMMARY: Key differences
    # ========================================
    print("\n" + "="*60)
    print("KEY LEARNING POINTS")
    print("="*60)
    print("- Different operation: addition -> multiplication")
    print("- Modified kernel function and file name")
    print("- Same parallel structure, different computation")

if __name__ == "__main__":
    sys.exit(main())
