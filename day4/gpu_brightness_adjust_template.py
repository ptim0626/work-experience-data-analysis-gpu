#!/usr/bin/env python3
"""
GPU brightness adjustment for 2D images - template.
"""
import cupy as cp
import numpy as np
from pathlib import Path
import sys
import time


def simulated_image_stack(num_images=10, height=512, width=512):
    """
    Create a stack of simulated images (with rings).

    Parameters
    ----------
    num_images : int, optional
        number of images in the stack. Default to 10.
    height : int, optional
        the height (number of rows) of the images. Default to 512.
    width : int, optional
        the width (number of columns) of the images. Default to 512.

    Returns
    -------
    stack : ndarray
        the stack of images as np.float32
    """
    print(f"Creating {num_images} images of size (height x width) "
          f"{height} x {width}...")
    stack = np.zeros((num_images, height, width), dtype=np.float32)

    # create centre coordinates
    centre_y, centre_x = height // 2, width // 2

    # create coordinate grids
    y, x = np.ogrid[:height, :width]

    rng = np.random.default_rng()
    for i in range(num_images):
        # vary parameters slightly for each image
        base_intensity = rng.uniform(30, 70)

        # add background
        stack[i] = base_intensity + rng.normal(0, 10, (height, width))

        # add diffraction rings
        for radius in [50, 100, 150, 200]:
            # vary radius slightly
            r = radius + rng.uniform(-5, 5)
            # create ring
            dist = np.sqrt((x - centre_x)**2 + (y - centre_y)**2)
            ring_mask = np.abs(dist - r) < 3
            ring_intensity = 100 + rng.uniform(0, 50)
            stack[i][ring_mask] += ring_intensity

    stack = stack.astype(np.float32)
    print(f"Number of bytes of the stack: {stack.size/2**30:.4f} GiB.")

    return stack

def cpu_brightness_adjustment(image_stack, brightness_value):
    """
    Adjust brightness of image stack using CPU (nested loops).

    This demonstrates the operation that will be parallelised on GPU.

    IMPORTANT
    =========
    For this specific operation, you would normally just do image_stack
    + brightness_value. I am cheating for the sake of performance
    comparison.

    Parameters
    ----------
    image_stack : ndarray
        the stack of image
    brightness_value : float
        the brightness value to be adjusted

    Returns
    -------
    output_stack : ndarray
        the adjusted stack of image
    """
    num_images, height, width = image_stack.shape
    output_stack = np.zeros_like(image_stack)

    # process each image
    for img_idx in range(num_images):
        # nested loops for each pixel - these become GPU threads
        for row in range(height):
            for col in range(width):
                # kernel operation: adjust brightness
                new_value = image_stack[img_idx, row, col] + brightness_value
                output_stack[img_idx, row, col] = new_value

    return output_stack

def gpu_brightness_adjustment(image_stack, brightness_value):
    """
    Adjust brightness of image stack using GPU.

    Parameters
    ----------
    image_stack : ndarray
        the stack of image
    brightness_value : float
        the brightness value to be adjusted

    Returns
    -------
    output_stack : ndarray
        the adjusted stack of image (in CPU memory)
    (compile_time, h2d_time, process_time, d2h_time) : tuple of float
        the time for compiling the kernel, host-to-device, actual
        calculation and device-to-host, respectively
    """
    num_images, height, width = image_stack.shape

    # compile the kernel
    kernel_file = Path("kernel_brightness_adjust.cu")
    if not kernel_file.exists():
        raise FileNotFoundError(
                f"Kernel file '{kernel_file}' not found! "
                "Make sure the .cu file is in the same directory as "
                "this script."
                )
    kernel_code = kernel_file.read_text()
    start_time = time.perf_counter()

    # TODO: create a CuPy RawKernel object from kernel_code
    # brightness_adjust_kernel = ...

    compile_time = time.perf_counter() - start_time

    # transfer data to gpu and initialise output
    start_time = time.perf_counter()

    # TODO: transfer the input image_stack to GPU memory
    # d_input_stack = ...

    # TODO: create an empty output array on GPU with same shape as input
    # d_output_stack = ...

    h2d_time = time.perf_counter() - start_time

    # define thread block and grid dimensions
    # using 16x16 thread blocks (common choice for 2D processing)
    threads_per_block = (16, 16)

    # TODO: calculate the number of blocks needed in x and y directions
    # remember that we need enough blocks to cover all pixels
    # if image width is 512 and we have 16 threads per block in x,
    # we need ceiling(512/16) = 32 blocks in x direction
    # blocks_per_grid_x = ...
    # blocks_per_grid_y = ...
    # blocks_per_grid = (..., ...)

    # ensure correct data types for kernel parameters
    height = cp.int32(height)
    width = cp.int32(width)
    brightness_value = cp.float32(brightness_value)

    # process each image in the stack
    start_time = time.perf_counter()
    for img_idx in range(num_images):
        # TODO: launch the kernel for this image
        # the kernel expects these parameters in order:
        # 1. input image (d_input_stack[img_idx])
        # 2. output image (d_output_stack[img_idx])
        # 3. height (as cp.int32)
        # 4. width (as cp.int32)
        # 5. brightness_value (as cp.float32)
        # brightness_adjust_kernel(blocks_per_grid, threads_per_block, ...)
        pass

    process_time = time.perf_counter() - start_time

    # transfer to CPU
    start_time = time.perf_counter()

    # TODO: transfer the GPU output back to CPU memory using .get()
    # output_stack = ...

    d2h_time = time.perf_counter() - start_time

    return output_stack, (compile_time, h2d_time, process_time, d2h_time)

def compare_performance(image_stack, brightness_value=30):
    """Compare CPU and GPU performance."""
    print(f"\nComparing performance for {image_stack.shape[0]} images "
          f"of size {image_stack.shape[1]}x{image_stack.shape[2]}")
    print(f"Brightness adjustment: +{brightness_value}")

    # time cpu version
    print("\nCPU processing...")
    start_time = time.perf_counter()
    cpu_result = cpu_brightness_adjustment(image_stack, brightness_value)
    cpu_time = time.perf_counter() - start_time
    print(f"CPU time: {cpu_time:.3f} s")

    # time gpu version
    print("\nGPU processing...")
    gpu_result, gpu_times = gpu_brightness_adjustment(image_stack,
                                                      brightness_value)
    print(f"GPU time:")
    print(f"    Total  : {sum(gpu_times):.3f} s")
    print(f"    Compile: {gpu_times[0]:.3f} s")
    print(f"    H2D    : {gpu_times[1]:.3f} s")
    print(f"    Process: {gpu_times[2]:.3f} s")
    print(f"    D2H    : {gpu_times[3]:.3f} s")

    # verify results match
    if np.allclose(cpu_result, gpu_result, rtol=1e-5):
        print("\nResults match between CPU and GPU.")
        print(f"Speed-up: {cpu_time/sum(gpu_times):.2f}x")
    else:
        print("\nResults do not match.")
        diff = np.abs(cpu_result - gpu_result)
        print(f"Maximum difference: {np.max(diff)}")

def main():
    # create simulated image stack
    image_stack = simulated_image_stack(num_images=5, height=512, width=512)

    # run performance comparison
    compare_performance(image_stack, brightness_value=40)

if __name__ == "__main__":
    sys.exit(main())
