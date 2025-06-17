#!/usr/bin/env python3
"""
GPU Hello World - Your First GPU Programme
==========================================
This programme demonstrates how to run code on the GPU using CuPy.
We'll add 10 to every element in an array - but on the GPU.

This script loads the CUDA kernel from a separate .cu file,
which allows for better syntax highlighting and code organisation.
"""

import cupy as cp
import numpy as np
from pathlib import Path
import sys

def main():
    print("="*60)
    print("GPU HELLO WORLD - ADDING 10 TO 1000 NUMBERS")
    print("="*60)

    # ========================================
    # STEP 1: Load the GPU kernel from file
    # ========================================
    print("\n[STEP 1] Loading CUDA kernel from file...")
    kernel_file = Path('kernel_add_constant.cu')

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
    # the second parameter 'add_constant' must match the function name
    # in .cu file
    add_kernel = cp.RawKernel(kernel_code, 'add_constant')
    print("Kernel compiled successfully")

    # ==========================================
    # STEP 2: Create test data on the CPU
    # ==========================================
    print("\n[STEP 2] Creating test data...")
    array_size = 1000
    cpu_data = np.arange(array_size, dtype=np.float32)
    # np.arange creates [0.0, 1.0, 2.0, ..., 999.0]
    # dtype=np.float32 uses 32-bit floats (standard for GPU computation)

    print(f"Created array with {array_size} elements")
    print(f"First 5 elements: {cpu_data[:5]}")
    print(f"Last 5 elements:  {cpu_data[-5:]}")

    # =============================================
    # STEP 3: Copy data from CPU to GPU memory
    # =============================================
    print("\n[STEP 3] Transferring data to GPU...")
    gpu_data = cp.asarray(cpu_data)
    # this copies our array from system RAM to GPU VRAM

    print("Data copied to GPU memory (host-to-device)")

    # ===============================================
    # STEP 4: Configure the GPU execution layout
    # ===============================================
    print("\n[STEP 4] Configuring GPU execution...")

    # how many threads should work together in one block?
    threads_per_block = 256
    # why 256?
    # - must be multiple of 32 (warp size - the GPU's basic execution unit)
    # - common choices: 128, 256, 512 (256 is a good default)
    # - too small = inefficient, too large = might exceed GPU limits

    # how many blocks do we need to cover all elements?
    blocks_per_grid = (array_size + threads_per_block - 1) // threads_per_block
    # let's break down this calculation:
    # - we have 1000 elements to process
    # - each block handles 256 elements
    # - 1000 / 256 = 3.90625 (we need 4 blocks!)
    # - the formula (1000 + 255) // 256 = 1255 // 256 = 4
    # - this ensures we always round UP to have enough threads

    total_threads = blocks_per_grid * threads_per_block
    print(f"GPU configuration:")
    print(f"Array size:        {array_size} elements")
    print(f"Threads per block: {threads_per_block}")
    print(f"Number of blocks:  {blocks_per_grid}")
    print(f"Total threads:     {total_threads}")
    print(f"Excess threads:    {total_threads - array_size} (will be idle)")

    # ===================================
    # STEP 5: Launch the kernel on GPU
    # ===================================
    print("\n[STEP 5] Launching GPU kernel...")
    print("All threads starting simultaneously.")

    # this is the moment - thousands of threads start at once!
    add_kernel(
        (blocks_per_grid,),      # grid size: how many blocks
        (threads_per_block,),    # block size: threads per block
        (gpu_data, array_size)   # arguments passed to kernel function
    )
    # note: the commas make these tuples, not just parentheses (dim3 struct)

    # CuPy automatically waits for the GPU to finish
    print("GPU computation completed")

    # =============================================
    # STEP 6: Copy results back and verify
    # =============================================
    print("\n[STEP 6] Retrieving results...")
    result = cp.asnumpy(gpu_data)
    # convert from CuPy GPU array back to NumPy CPU array

    print("Results copied back to CPU (device-to-host)")
    print(f"First 5 elements: {result[:5]}")
    print(f"Last 5 elements:  {result[-5:]}")

    # verify correctness
    print("\n[VERIFICATION] Checking results...")
    expected = cpu_data + 10
    errors = np.abs(result - expected)
    max_error = np.max(errors)

    if np.allclose(result, expected):
        print("SUCCESS! All values correctly increased by 10")
    else:
        print("ERROR: Results don't match expected values")
    print(f"Maximum error: {max_error}")

    # ========================================
    # SUMMARY: What just happened?
    # ========================================
    print("\n" + "="*60)
    print("WHAT JUST HAPPENED?")
    print("="*60)
    print(f"- {total_threads} threads launched simultaneously")
    print(f"- Each thread handled 1 element (parallel processing)")
    print(f"- CPU would process these one-by-one (serial processing)")
    print(f"- For larger arrays, GPU advantage becomes even greater!")
    print("\nNext steps: Try modifying the kernel to add a different")
    print("value, or experiment with different array sizes!")

    # memory cleanup happens automatically when programme ends
    # CuPy handles freeing both CPU and GPU memory

if __name__ == "__main__":
    sys.exit(main())
