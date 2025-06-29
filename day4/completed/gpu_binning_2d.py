#!/usr/bin/env python3
"""
GPU pixel binning for 2D images.

This script shows the complete implementation of GPU kernels for pixel
binning using CuPy, with CPU comparison.
"""
import cupy as cp
import numpy as np
from pathlib import Path
import sys
import time


def simulated_detector_image_stack(num_images=10, height=512, width=512):
    """
    Create a stack of simulated detector images.

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
        the stack of detector images as np.float32
    """
    print(f"Creating {num_images} detector images of size (height x width) "
          f"{height} x {width}...")
    stack = np.zeros((num_images, height, width), dtype=np.float32)

    rng = np.random.default_rng()

    for i in range(num_images):
        # create background with poisson noise
        background = rng.poisson(lam=5, size=(height, width)).astype(
            np.float32)

        # create diffraction rings
        y, x = np.ogrid[:height, :width]
        center_y, center_x = height // 2, width // 2
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)

        # add rings at different radii
        signal = background.copy()
        ring_radii = (height * 0.15, height * 0.25, height * 0.35)
        ring_width = height * 0.03

        for radius in ring_radii:
            # create ring mask
            ring_mask = (r >= radius - ring_width/2) & (
                r <= radius + ring_width/2)
            # add photons to ring
            ring_photons = rng.poisson(lam=50, size=signal.shape)
            signal[ring_mask] += ring_photons[ring_mask].astype(
                np.float32)

        stack[i] = signal

    stack = stack.astype(np.float32)
    print(f"Number of bytes of the stack: {stack.nbytes/2**30:.4f} GiB.")

    return stack


def cpu_bin_pixels(image_stack, bin_size=2):
    """
    Apply pixel binning to image stack using CPU (nested loops).

    This demonstrates the operation that will be parallelised on GPU.

    Parameters
    ----------
    image_stack : ndarray
        the stack of images
    bin_size : int
        size of the square bins

    Returns
    -------
    output_stack : ndarray
        the binned stack of images
    """
    # validate input
    if bin_size <= 0:
        raise ValueError("bin_size must be positive.")

    num_images, height, width = image_stack.shape

    # calculate output dimensions (round up for partial bins)
    out_height = (height + bin_size - 1) // bin_size
    out_width = (width + bin_size - 1) // bin_size

    output_stack = np.zeros((num_images, out_height, out_width),
                            dtype=np.float32)

    # process each image
    for img_idx in range(num_images):
        # process each output pixel (bin)
        for bin_row in range(out_height):
            for bin_col in range(out_width):
                # sum pixels belonging to this bin
                bin_sum = np.float32(0.0)

                for dy in range(bin_size):
                    for dx in range(bin_size):
                        # calculate source pixel position
                        src_row = bin_row * bin_size + dy
                        src_col = bin_col * bin_size + dx

                        # boundary check
                        if src_row < height and src_col < width:
                            bin_sum += image_stack[img_idx, src_row, src_col]

                # store result for this bin
                output_stack[img_idx, bin_row, bin_col] = bin_sum

    return output_stack


def gpu_bin_pixels(image_stack, bin_size=2):
    """
    Apply pixel binning to image stack using GPU.

    Parameters
    ----------
    image_stack : ndarray
        the stack of images
    bin_size : int
        size of the square bins

    Returns
    -------
    output_stack : ndarray
        the binned stack of images (in CPU memory)
    (compile_time, h2d_time, process_time, d2h_time) : tuple of float
        the time for compiling the kernel, host-to-device, actual
        calculation and device-to-host, respectively
    """
    # validate input
    if bin_size <= 0:
        raise ValueError("bin_size must be positive.")

    num_images, height, width = image_stack.shape

    # calculate output dimensions
    out_height = (height + bin_size - 1) // bin_size
    out_width = (width + bin_size - 1) // bin_size

    # compile the kernel
    kernel_file = Path("kernel_binning_2d.cu")
    if not kernel_file.exists():
        raise FileNotFoundError(
                f"Kernel file '{kernel_file}' not found! "
                "Make sure the .cu file is in the same directory as "
                "this script."
                )
    kernel_code = kernel_file.read_text()
    start_time = time.perf_counter()
    binning_kernel = cp.RawKernel(kernel_code, "pixel_binning_2d")
    compile_time = time.perf_counter() - start_time

    # transfer data to gpu and initialise output
    start_time = time.perf_counter()
    d_input_stack = cp.asarray(image_stack)
    d_output_stack = cp.zeros((num_images, out_height, out_width),
                              dtype=cp.float32)
    h2d_time = time.perf_counter() - start_time

    # define thread block and grid dimensions
    threads_per_block = (16, 16)
    blocks_per_grid_x = ((out_width + threads_per_block[0] - 1) //
                          threads_per_block[0])
    blocks_per_grid_y = ((out_height + threads_per_block[1] - 1) //
                          threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # ensure correct data types
    in_height = cp.int32(height)
    in_width = cp.int32(width)
    bin_size = cp.int32(bin_size)

    # process each image in the stack
    start_time = time.perf_counter()
    for img_idx in range(num_images):
        # launch kernel for this image
        binning_kernel(blocks_per_grid,
                       threads_per_block,
                       (d_input_stack[img_idx],
                        d_output_stack[img_idx],
                        in_height, in_width, bin_size)
                       )
    process_time = time.perf_counter() - start_time

    # transfer to CPU
    start_time = time.perf_counter()
    output_stack = d_output_stack.get()
    d2h_time = time.perf_counter() - start_time

    return output_stack, (compile_time, h2d_time, process_time, d2h_time)


def compare_performance(image_stack, bin_size=2):
    """Compare CPU and GPU performance for pixel binning."""
    print(f"\nComparing performance for {image_stack.shape[0]} images "
          f"of size {image_stack.shape[1]}x{image_stack.shape[2]}")
    print(f"Binning size: {bin_size}x{bin_size}")

    # time cpu version
    print("\nCPU processing (sum mode)...")
    start_time = time.perf_counter()
    cpu_result = cpu_bin_pixels(image_stack, bin_size)
    cpu_time = time.perf_counter() - start_time
    print(f"CPU time: {cpu_time:.3f} s")
    print(f"Output size: {cpu_result.shape[1]}x{cpu_result.shape[2]}")

    # time gpu version
    print("\nGPU processing (sum mode)...")
    gpu_result, gpu_times = gpu_bin_pixels(image_stack, bin_size)
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
        print(f"Speed-up (processing only): {cpu_time/gpu_times[2]:.2f}x")
    else:
        print("\nResults do not match.")
        diff = np.abs(cpu_result - gpu_result)
        print(f"Maximum difference: {np.max(diff)}")
        print(f"Mean difference: {np.mean(diff)}")

    # check intensity preservation
    total_input = np.sum(image_stack[0])
    total_output_cpu = np.sum(cpu_result[0])
    total_output_gpu = np.sum(gpu_result[0])
    print(f"\nIntensity preservation (first image):")
    print(f"    Input : {total_input:.0f}")
    print(f"    CPU   : {total_output_cpu:.0f}")
    print(f"    GPU   : {total_output_gpu:.0f}")


def main():
    # create simulated detector image stack
    image_stack = simulated_detector_image_stack(num_images=5,
                                                 height=1024,
                                                 width=1024
                                                 )

    # run performance comparison
    compare_performance(image_stack, bin_size=4)


if __name__ == "__main__":
    sys.exit(main())
