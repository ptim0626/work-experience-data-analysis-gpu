#!/usr/bin/env python3
"""
GPU median filter for 2D images - template.
"""
import cupy as cp
import numpy as np
from pathlib import Path
import sys
import time


def simulated_noisy_image_stack(num_images=10, height=512, width=512,
                                noise_type="salt_pepper"):
    """
    Create a stack of simulated noisy images.

    Parameters
    ----------
    num_images : int, optional
        number of images in the stack. Default to 10.
    height : int, optional
        the height (number of rows) of the images. Default to 512.
    width : int, optional
        the width (number of columns) of the images. Default to 512.
    noise_type : str, optional
        type of noise to add. "salt_pepper" or "gaussian". Default to
        "salt_pepper".

    Returns
    -------
    stack : ndarray
        the stack of noisy images as np.float32
    """
    print(f"Creating {num_images} noisy images of size (height x width) "
          f"{height} x {width}...")
    stack = np.zeros((num_images, height, width), dtype=np.float32)

    rng = np.random.default_rng()

    for i in range(num_images):
        # create base image with gradient pattern
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        xx, yy = np.meshgrid(x, y)

        # create smooth pattern
        base_image = (np.sin(5 * xx) * np.cos(3 * yy) + 1) * 127.5

        if noise_type == "salt_pepper":
            # add salt and pepper noise
            noisy_image = base_image.copy()

            # salt (white spots)
            salt_mask = rng.random((height, width)) < 0.05
            noisy_image[salt_mask] = 255

            # pepper (black spots)
            pepper_mask = rng.random((height, width)) < 0.05
            noisy_image[pepper_mask] = 0

            stack[i] = noisy_image
        else:
            # add gaussian noise
            noise = rng.normal(0, 20, (height, width))
            stack[i] = np.clip(base_image + noise, 0, 255)

    stack = stack.astype(np.float32)
    print(f"Number of bytes of the stack: {stack.nbytes/2**30:.4f} GiB.")

    return stack


def calculate_median_explicit(window):
    """
    Calculate median using explicit bubble sort.

    This matches the CPU implementation from the notebook.

    Parameters
    ----------
    window : ndarray
        flattened array of neighbourhood values

    Returns
    -------
    median : float
        the median value
    """
    # make a copy to avoid modifying original
    sorted_window = window.copy()
    n = len(sorted_window)

    # bubble sort
    for i in range(n):
        for j in range(0, n - i - 1):
            if sorted_window[j] > sorted_window[j + 1]:
                # swap elements
                temp = sorted_window[j]
                sorted_window[j] = sorted_window[j + 1]
                sorted_window[j + 1] = temp

    # find middle element
    middle_index = n // 2
    return sorted_window[middle_index]


def cpu_median_filter(image_stack, window_size=3):
    """
    Apply median filter to image stack using CPU (nested loops).

    This demonstrates the operation that will be parallelised on GPU.
    Uses "valid" mode - output is smaller than input.

    Parameters
    ----------
    image_stack : ndarray
        the stack of images
    window_size : int
        size of the filter window (must be odd)

    Returns
    -------
    output_stack : ndarray
        the filtered stack of images
    """
    # ensure window size is odd
    if window_size % 2 == 0:
        window_size += 1
        print(f"Window size must be odd, using {window_size}")

    num_images, height, width = image_stack.shape
    half_window = window_size // 2

    # output is smaller because of "valid" mode
    out_height = height - 2 * half_window
    out_width = width - 2 * half_window
    output_stack = np.zeros((num_images, out_height, out_width),
                            dtype=np.float32)

    # process each image
    for img_idx in range(num_images):
        # process each valid pixel
        for row in range(half_window, height - half_window):
            for col in range(half_window, width - half_window):
                # extract neighbourhood into flat array
                window = np.zeros(window_size * window_size,
                                  dtype=np.float32)
                idx = 0

                for wr in range(-half_window, half_window + 1):
                    for wc in range(-half_window, half_window + 1):
                        window[idx] = image_stack[img_idx, row + wr,
                                                  col + wc]
                        idx += 1

                # calculate median
                median_val = calculate_median_explicit(window)

                # store in output (adjust indices for smaller size)
                output_stack[img_idx, row - half_window,
                             col - half_window] = median_val

    return output_stack


def gpu_median_filter(image_stack, window_size=3):
    """
    Apply median filter to image stack using GPU.

    Uses "valid" mode - output is smaller than input.

    Parameters
    ----------
    image_stack : ndarray
        the stack of images
    window_size : int
        size of the filter window (must be odd)

    Returns
    -------
    output_stack : ndarray
        the filtered stack of images (in CPU memory)
    (compile_time, h2d_time, process_time, d2h_time) : tuple of float
        the time for compiling the kernel, host-to-device, actual
        calculation and device-to-host, respectively
    """
    # ensure window size is odd
    if window_size % 2 == 0:
        window_size += 1
        print(f"Window size must be odd, using {window_size}")

    # check window size limit
    if window_size > 9:
        raise ValueError(
            "Window size must be 9 or smaller (kernel limitation)")

    num_images, height, width = image_stack.shape
    half_window = window_size // 2

    # calculate output dimensions
    out_height = height - 2 * half_window
    out_width = width - 2 * half_window

    # compile the kernel
    kernel_file = Path("kernel_median_filter_2d.cu")
    if not kernel_file.exists():
        raise FileNotFoundError(
                f"Kernel file '{kernel_file}' not found! "
                "Make sure the .cu file is in the same directory as "
                "this script."
                )
    kernel_code = kernel_file.read_text()
    start_time = time.perf_counter()

    # TODO: create a CuPy RawKernel object from kernel_code
    # median_filter_kernel = ...

    compile_time = time.perf_counter() - start_time

    # transfer data to gpu and initialise output
    start_time = time.perf_counter()
    d_input_stack = cp.asarray(image_stack)

    # TODO: create output array on GPU with correct dimensions
    # d_output_stack = ...

    h2d_time = time.perf_counter() - start_time

    # define thread block and grid dimensions
    threads_per_block = (16, 16)

    # TODO: calculate the number of blocks needed in x and y directions
    # blocks_per_grid_x = ...
    # blocks_per_grid_y = ...
    # blocks_per_grid = (..., ...)

    # ensure correct data types
    # TODO: convert parameters to correct GPU data types
    # in_height = ...
    # in_width = ...
    # window_size = ...

    # process each image in the stack
    start_time = time.perf_counter()
    for img_idx in range(num_images):
        # TODO: launch the kernel for this image
        # the kernel expects these parameters in order:
        # 1. Input image (d_input_stack[img_idx])
        # 2. Output image (d_output_stack[img_idx])
        # 3. Input height (in_height)
        # 4. Input width (in_width)
        # 5. Window size (window_size)
        pass
    process_time = time.perf_counter() - start_time

    # transfer to CPU
    start_time = time.perf_counter()
    output_stack = d_output_stack.get()
    d2h_time = time.perf_counter() - start_time

    return output_stack, (compile_time, h2d_time, process_time, d2h_time)


def cpu_median_filter_padded(image_stack, window_size=3, padding="edge"):
    """
    Apply median filter with padding to image stack using CPU.

    Maintains same size as input by applying padding.

    Parameters
    ----------
    image_stack : ndarray
        the stack of images
    window_size : int
        size of the filter window (must be odd)
    padding : str
        padding mode: "zero", "edge", or "reflect"

    Returns
    -------
    output_stack : ndarray
        the filtered stack of images (same size as input)
    """
    # ensure window size is odd
    if window_size % 2 == 0:
        window_size += 1
        print(f"Window size must be odd, using {window_size}")

    num_images, height, width = image_stack.shape
    half_window = window_size // 2

    # output has same size as input
    output_stack = np.zeros_like(image_stack)

    # process each image
    for img_idx in range(num_images):
        image = image_stack[img_idx]

        # apply padding to current image
        if padding == "edge":
            # repeat edge values
            padded = np.pad(image, half_window, mode="edge")
        elif padding == "reflect":
            # mirror values at edges
            padded = np.pad(image, half_window, mode="reflect")
        elif padding == "zero":
            # pad with zeros
            padded = np.pad(image, half_window, mode="constant",
                           constant_values=0)
        else:
            raise ValueError(f"Unsupported padding strategy '{padding}'.")

        # now process all pixels in the original image
        for row in range(height):
            for col in range(width):
                # extract window from padded image
                window = np.zeros(window_size * window_size, dtype=np.float32)
                idx = 0

                for wr in range(window_size):
                    for wc in range(window_size):
                        # adjust for padding offset
                        window[idx] = padded[row + wr, col + wc]
                        idx += 1

                # calculate median using explicit operations
                median_val = calculate_median_explicit(window)

                # store in output
                output_stack[img_idx, row, col] = median_val

    return output_stack


def gpu_median_filter_padded(image_stack, window_size=3, padding="edge"):
    """
    Apply median filter with padding to maintain image size.

    Parameters
    ----------
    image_stack : ndarray
        the stack of images
    window_size : int
        size of the filter window (must be odd)
    padding : str
        padding mode: "zero", "edge", or "reflect"

    Returns
    -------
    output_stack : ndarray
        the filtered stack of images (same size as input)
    gpu_times : tuple of float
        timing information
    """
    # ensure window size is odd
    if window_size % 2 == 0:
        window_size += 1
        print(f"Window size must be odd, using {window_size}")

    # check window size limit
    if window_size > 9:
        raise ValueError("Window size must be 9 or smaller (kernel limitation)")

    num_images, height, width = image_stack.shape

    # map padding mode to integer
    padding_modes = {"zero": 0, "edge": 1, "reflect": 2}
    if padding not in padding_modes:
        raise ValueError(f"Unknown padding mode: {padding}")
    padding_mode = padding_modes[padding]

    # compile the kernel
    kernel_file = Path("kernel_median_filter_2d.cu")
    if not kernel_file.exists():
        raise FileNotFoundError(
                f"Kernel file '{kernel_file}' not found! "
                "Make sure the .cu file is in the same directory as "
                "this script."
                )
    kernel_code = kernel_file.read_text()
    start_time = time.perf_counter()

    # TODO: create a CuPy RawKernel for the padded version
    # use kernel function name "median_filter_2d_padded"
    # median_filter_kernel = ...

    compile_time = time.perf_counter() - start_time

    # transfer data to gpu and initialise output
    start_time = time.perf_counter()
    d_input_stack = cp.asarray(image_stack)

    # TODO: create output array with same size as input
    # d_output_stack = ...

    h2d_time = time.perf_counter() - start_time

    # define thread block and grid dimensions
    threads_per_block = (16, 16)

    # TODO: calculate blocks per grid
    # blocks_per_grid_x = ...
    # blocks_per_grid_y = ...
    # blocks_per_grid = ...

    # ensure correct data types
    # TODO: convert all parameters to GPU data types
    # height = ...
    # width = ...
    # window_size = ...
    # padding_mode = ...

    # process each image in the stack
    start_time = time.perf_counter()
    for img_idx in range(num_images):
        # TODO: launch the padded kernel
        # the kernel expects these parameters in order:
        # 1. Input image
        # 2. Output image
        # 3. Height
        # 4. Width
        # 5. Window size
        # 6. Padding mode
        pass
    process_time = time.perf_counter() - start_time

    # transfer to CPU
    start_time = time.perf_counter()
    output_stack = d_output_stack.get()
    d2h_time = time.perf_counter() - start_time

    return output_stack, (compile_time, h2d_time, process_time, d2h_time)


def compare_performance(image_stack, window_size=3):
    """Compare CPU and GPU performance for median filtering."""
    print(f"\nComparing performance for {image_stack.shape[0]} images "
          f"of size {image_stack.shape[1]}x{image_stack.shape[2]}")
    print(f"Median filter window size: {window_size}x{window_size}")

    # time cpu version
    print("\nCPU processing (valid mode)...")
    start_time = time.perf_counter()
    cpu_result = cpu_median_filter(image_stack, window_size)
    cpu_time = time.perf_counter() - start_time
    print(f"CPU time: {cpu_time:.3f} s")
    print(f"Output size: {cpu_result.shape[1]}x{cpu_result.shape[2]}")

    # time gpu version
    print("\nGPU processing (valid mode)...")
    gpu_result, gpu_times = gpu_median_filter(image_stack, window_size)
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


def compare_performance_padding(image_stack, window_size=3, padding="edge"):
    """Compare CPU and GPU performance for median filtering with padding."""
    print(f"\nComparing performance for {image_stack.shape[0]} images "
          f"of size {image_stack.shape[1]}x{image_stack.shape[2]}")
    print(f"Median filter window size: {window_size}x{window_size}")

    # time cpu version
    print("\nCPU processing (with padding)...")
    start_time = time.perf_counter()
    cpu_result = cpu_median_filter_padded(image_stack, window_size, padding)
    cpu_time = time.perf_counter() - start_time
    print(f"CPU time: {cpu_time:.3f} s")
    print(f"Output size: {cpu_result.shape[1]}x{cpu_result.shape[2]}")

    # time gpu version
    print("\nGPU processing (with padding)...")
    gpu_result, gpu_times = gpu_median_filter_padded(image_stack,
                                                     window_size,
                                                     padding
                                                     )
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


def main():
    # create simulated noisy image stack
    image_stack = simulated_noisy_image_stack(num_images=5,
                                              height=512,
                                              width=512,
                                              noise_type="salt_pepper")
    # window size used (assume maximum window size is 9)
    ws = 3

    # run performance comparison
    compare_performance(image_stack, window_size=ws)

    # with padding
    print("="*60)
    compare_performance_padding(image_stack, window_size=ws, padding="edge")


if __name__ == "__main__":
    sys.exit(main())
