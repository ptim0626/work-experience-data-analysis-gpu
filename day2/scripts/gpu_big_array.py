#!/usr/bin/env python3
"""
GPU Big Array - Demonstrating GPU Performance Benefits
====================================================
This programme processes 100 million elements and compares
GPU vs CPU performance to show when GPUs become beneficial.
"""

import numpy as np
import cupy as cp
from pathlib import Path
import sys
import time

def main():
    print("="*60)
    print("GPU BIG ARRAY - ADDING 10 TO 100 MILLION NUMBERS")
    print("="*60)

    # ========================================
    # STEP 1: Load the GPU kernel from file
    # ========================================
    print("\n[STEP 1] Loading CUDA kernel from file...")
    kernel_file = Path("kernel_add_constant.cu")

    if not kernel_file.exists():
        raise FileNotFoundError(
            f"Kernel file '{kernel_file}' not found! "
            "Make sure the .cu file is in the same directory as this script."
        )

    kernel_code = kernel_file.read_text()
    print(f"Successfully loaded kernel from {kernel_file}")

    add_kernel = cp.RawKernel(kernel_code, 'add_constant')
    print("Kernel compiled successfully")

    # ==========================================
    # STEP 2: Create test data on the CPU
    # ==========================================
    print("\n[STEP 2] Creating test data...")
    # much larger array size (100 million elements)
    array_size = int(1e8)
    cpu_data = np.arange(array_size, dtype=np.float32)

    print(f"Created array with {array_size:,} elements")
    print(f"Memory size: {cpu_data.nbytes / (2**20):.2f} MB")
    print(f"First 5 elements: {cpu_data[:5]}")
    print(f"Last 5 elements:  {cpu_data[-5:]}")

    # =============================================
    # STEP 3: CPU timing - traditional approach
    # =============================================
    print("\n[STEP 3] CPU computation...")
    print("Processing elements one by one...")

    # make a copy for CPU processing
    cpu_result = cpu_data.copy()

    # time the CPU version
    cpu_start = time.time()
    # traditional loop - what GPU replaces
    # not using NumPy vectorisation for demo purpose
    for i in range(array_size):
        cpu_result[i] = cpu_result[i] + 10.0
    cpu_end = time.time()
    cpu_time = cpu_end - cpu_start

    print(f"CPU completed in {cpu_time:.4f} seconds")

    # =============================================
    # STEP 4: GPU computation with timing
    # =============================================
    print("\n[STEP 4] GPU computation...")

    # copy data to GPU
    print("Transferring data to GPU...")
    transfer_start = time.time()
    gpu_data = cp.asarray(cpu_data)
    transfer_time = time.time() - transfer_start
    print(f"Transfer time: {transfer_time:.4f} seconds")

    # configure GPU execution
    threads_per_block = 256
    blocks_per_grid = (array_size + threads_per_block - 1) // threads_per_block

    print(f"\nGPU configuration:")
    print(f"Array size:        {array_size:,} elements")
    print(f"Threads per block: {threads_per_block}")
    print(f"Number of blocks:  {blocks_per_grid:,}")
    print(f"Total threads:     {blocks_per_grid * threads_per_block:,}")

    # launch kernel with timing
    print("\nLaunching kernel...")
    # ensure previous operations complete
    cp.cuda.Stream.null.synchronize()

    kernel_start = time.time()
    add_kernel(
        (blocks_per_grid,),
        (threads_per_block,),
        (gpu_data, array_size)
    )
    # wait for kernel to complete
    cp.cuda.Stream.null.synchronize()
    kernel_end = time.time()
    kernel_time = kernel_end - kernel_start

    print(f"Kernel execution time: {kernel_time:.4f} seconds")

    # copy results back
    print("\nTransferring results back...")
    transfer_back_start = time.time()
    gpu_result = cp.asnumpy(gpu_data)
    transfer_back_time = time.time() - transfer_back_start
    print(f"Transfer back time: {transfer_back_time:.4f} seconds")

    # total GPU time
    gpu_total_time = transfer_time + kernel_time + transfer_back_time
    print(f"\nTotal GPU time: {gpu_total_time:.4f} seconds")

    # =============================================
    # STEP 5: Performance comparison
    # =============================================
    print("\n[PERFORMANCE COMPARISON]")
    print("="*50)
    print(f"CPU time:         {cpu_time:.4f} seconds")
    print(f"GPU total time:   {gpu_total_time:.4f} seconds")
    print(f"GPU kernel only:  {kernel_time:.4f} seconds")
    print("="*50)

    # calculate speedup
    overall_speedup = cpu_time / gpu_total_time
    kernel_speedup = cpu_time / kernel_time

    print(f"Overall speedup:      {overall_speedup:.1f}x faster")
    print(f"Kernel-only speedup:  {kernel_speedup:.1f}x faster")

    # verify correctness
    print("\n[VERIFICATION] Checking results...")
    if np.allclose(gpu_result, cpu_result):
        print("SUCCESS! GPU and CPU results match")
    else:
        print("ERROR: Results don't match")

    print("\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)
    print("- With 100M elements, GPU shows significant speedup")
    print("- Data transfer time is overhead for GPU")
    print("- Kernel execution is much faster than CPU loop")
    print("- Larger arrays = better GPU efficiency")

if __name__ == "__main__":
    sys.exit(main())
