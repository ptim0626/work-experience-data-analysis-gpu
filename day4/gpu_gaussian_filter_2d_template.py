#!/usr/bin/env python3
"""
GPU Gaussian filter for 2D images - template.
"""
import cupy as cp
import numpy as np
from pathlib import Path
import sys
import time


def simulated_noisy_image_stack(num_images=10, height=512, width=512,
                                noise_type="gaussian"):
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
        type of noise to add. "gaussian" or "salt_pepper". Default to
        "gaussian".

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

        if noise_type == "gaussian":
            # add gaussian noise
            noise = rng.normal(0, 20, (height, width))
            stack[i] = np.clip(base_image + noise, 0, 255)
        else:
            # add salt and pepper noise
            noisy_image = base_image.copy()

            # salt (white spots)
            salt_mask = rng.random((height, width)) < 0.05
            noisy_image[salt_mask] = 255

            # pepper (black spots)
            pepper_mask = rng.random((height, width)) < 0.05
            noisy_image[pepper_mask] = 0

            stack[i] = noisy_image

    stack = stack.astype(np.float32)
    print(f"Number of bytes of the stack: {stack.size/2**30:.4f} GiB.")

    return stack


def create_gaussian_kernel_2d(sigma, kernel_size=None):
    """
    Create 2D Gaussian kernel weights.

    Parameters
    ----------
    sigma : float
        standard deviation of the Gaussian
    kernel_size : int or None
        size of the kernel. If None, automatically determine based on
        sigma

    Returns
    -------
    kernel : ndarray
        2D array of Gaussian weights, normalised to sum to 1
    """
    # input validation
    if sigma <= 0:
        raise ValueError("Sigma must be positive")

    # automatically determine kernel size if not provided
    if kernel_size is None:
        # cover 6-sigma range
        kernel_size = int(6 * sigma + 1)
        # ensure odd size for symmetry
        if kernel_size % 2 == 0:
            kernel_size += 1

    # validate kernel size
    if kernel_size < 1:
        raise ValueError("Kernel size must be at least 1.")
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd for symmetry.")

    # create position arrays centred at 0
    half_size = kernel_size // 2
    x = np.arange(-half_size, half_size + 1)
    y = np.arange(-half_size, half_size + 1)
    xx, yy = np.meshgrid(x, y)

    # calculate Gaussian weights (without the normalisation coefficient)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))

    # normalise so weights sum to 1
    kernel = kernel / kernel.sum()

    return kernel.astype(np.float32)


def cpu_gaussian_filter(image_stack, sigma=1.0, kernel_size=None):
    """
    Apply Gaussian filter to image stack using CPU (nested loops).

    This demonstrates the operation that will be parallelised on GPU.
    Uses "valid" mode - output is smaller than input.

    Parameters
    ----------
    image_stack : ndarray
        the stack of images
    sigma : float
        standard deviation of the Gaussian kernel
    kernel_size : int or None
        size of the filter kernel. If None, automatically determined

    Returns
    -------
    output_stack : ndarray
        the filtered stack of images
    """
    # create gaussian kernel
    kernel = create_gaussian_kernel_2d(sigma, kernel_size)
    kernel_size = kernel.shape[0]
    half_kernel = kernel_size // 2

    num_images, height, width = image_stack.shape

    # output is smaller because of "valid" mode
    out_height = height - 2 * half_kernel
    out_width = width - 2 * half_kernel
    output_stack = np.zeros((num_images, out_height, out_width),
                            dtype=np.float32)

    # process each image
    for img_idx in range(num_images):
        # process each valid pixel
        for row in range(half_kernel, height - half_kernel):
            for col in range(half_kernel, width - half_kernel):
                # accumulate weighted sum
                weighted_sum = 0.0

                # apply kernel weights - this is convolution
                for kr in range(kernel_size):
                    for kc in range(kernel_size):
                        # calculate position in input image
                        src_row = row + kr - half_kernel
                        src_col = col + kc - half_kernel

                        # multiply and accumulate
                        pixel_value = image_stack[img_idx, src_row, src_col]
                        kernel_weight = kernel[kr, kc]
                        weighted_sum += pixel_value * kernel_weight

                # store in output (adjust indices for smaller size)
                output_stack[img_idx, row - half_kernel,
                             col - half_kernel] = weighted_sum

    return output_stack


def gpu_gaussian_filter(image_stack, sigma=1.0, kernel_size=None):
    """
    Apply Gaussian filter to image stack using GPU.

    Uses "valid" mode - output is smaller than input.

    Parameters
    ----------
    image_stack : ndarray
        the stack of images
    sigma : float
        standard deviation of the Gaussian kernel
    kernel_size : int or None
        size of the filter kernel. If None, automatically determined

    Returns
    -------
    output_stack : ndarray
        the filtered stack of images (in CPU memory)
    (compile_time, h2d_time, process_time, d2h_time) : tuple of float
        the time for compiling the kernel, host-to-device, actual
        calculation and device-to-host, respectively
    """
    # create gaussian kernel on cpu
    kernel = create_gaussian_kernel_2d(sigma, kernel_size)
    kernel_size = kernel.shape[0]
    half_kernel = kernel_size // 2

    num_images, height, width = image_stack.shape

    # calculate output dimensions
    out_height = height - 2 * half_kernel
    out_width = width - 2 * half_kernel

    # compile the kernel
    kernel_file = Path("kernel_gaussian_filter_2d.cu")
    if not kernel_file.exists():
        raise FileNotFoundError(
                f"Kernel file '{kernel_file}' not found! "
                "Make sure the .cu file is in the same directory as "
                "this script."
                )
    kernel_code = kernel_file.read_text()
    start_time = time.perf_counter()

    # TODO: create a CuPy RawKernel object from kernel_code
    # use kernel function name "gaussian_filter_2d"
    # gaussian_filter_kernel = ...

    compile_time = time.perf_counter() - start_time

    # transfer data to gpu and initialise output
    start_time = time.perf_counter()
    d_input_stack = cp.asarray(image_stack)

    # TODO: create output array on GPU with correct dimensions
    # d_output_stack = ...

    # TODO: transfer the flattened kernel weights to GPU
    # Remember to flatten the 2D kernel array
    # d_kernel_weights = ...

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
    # kernel_size = ...

    # process each image in the stack
    start_time = time.perf_counter()
    for img_idx in range(num_images):
        # TODO: launch the kernel for this image
        # the kernel expects these parameters in order:
        # 1. Input image (d_input_stack[img_idx])
        # 2. Output image (d_output_stack[img_idx])
        # 3. Kernel weights (d_kernel_weights)
        # 4. Input height (in_height)
        # 5. Input width (in_width)
        # 6. Kernel size (kernel_size_gpu)
        pass
    process_time = time.perf_counter() - start_time

    # transfer to CPU
    start_time = time.perf_counter()
    output_stack = d_output_stack.get()
    d2h_time = time.perf_counter() - start_time

    return output_stack, (compile_time, h2d_time, process_time, d2h_time)


def cpu_gaussian_filter_padded(image_stack, sigma=1.0, kernel_size=None,
                               padding="edge"):
    """
    Apply Gaussian filter with padding to image stack using CPU.

    Maintains same size as input by applying padding.

    Parameters
    ----------
    image_stack : ndarray
        the stack of images
    sigma : float
        standard deviation of the Gaussian kernel
    kernel_size : int or None
        size of the filter kernel. If None, automatically determined
    padding : str
        padding mode: "zero", "edge", or "reflect"

    Returns
    -------
    output_stack : ndarray
        the filtered stack of images (same size as input)
    """
    # create gaussian kernel
    kernel = create_gaussian_kernel_2d(sigma, kernel_size)
    kernel_size = kernel.shape[0]
    half_kernel = kernel_size // 2

    num_images, height, width = image_stack.shape

    # output has same size as input
    output_stack = np.zeros_like(image_stack)

    # process each image
    for img_idx in range(num_images):
        # process all pixels
        for row in range(height):
            for col in range(width):
                # accumulate weighted sum and weight sum
                weighted_sum = 0.0
                weight_sum = 0.0

                # apply kernel weights
                for kr in range(kernel_size):
                    for kc in range(kernel_size):
                        # calculate position in input image
                        src_row = row + kr - half_kernel
                        src_col = col + kc - half_kernel

                        # handle boundaries based on mode
                        if padding == "zero":
                            # zero padding: use 0 for out-of-bounds
                            if (0 <= src_row < height and
                                0 <= src_col < width):
                                value = image_stack[img_idx, src_row, src_col]
                            else:
                                value = 0.0
                            weight = kernel[kr, kc]
                        elif padding == "reflect":
                            # reflect at boundaries
                            ref_row = src_row
                            ref_col = src_col
                            if ref_row < 0:
                                ref_row = -ref_row
                            elif ref_row >= height:
                                ref_row = 2 * height - ref_row - 2
                            if ref_col < 0:
                                ref_col = -ref_col
                            elif ref_col >= width:
                                ref_col = 2 * width - ref_col - 2
                            value = image_stack[img_idx, ref_row, ref_col]
                            weight = kernel[kr, kc]
                        else:
                            # constant (use edge value)
                            edge_row = max(0, min(height - 1, src_row))
                            edge_col = max(0, min(width - 1, src_col))
                            value = image_stack[img_idx, edge_row, edge_col]
                            weight = kernel[kr, kc]

                        if weight > 0:
                            weighted_sum += value * weight
                            weight_sum += weight

                # normalise by actual weight sum
                if weight_sum > 0:
                    output_stack[img_idx, row, col] = (weighted_sum /
                                                       weight_sum)
                else:
                    output_stack[img_idx, row, col] = 0

    return output_stack


def gpu_gaussian_filter_padded(image_stack, sigma=1.0, kernel_size=None,
                               padding="edge"):
    """
    Apply Gaussian filter with padding to maintain image size.

    Parameters
    ----------
    image_stack : ndarray
        the stack of images
    sigma : float
        standard deviation of the Gaussian kernel
    kernel_size : int or None
        size of the filter kernel. If None, automatically determined
    padding : str
        padding mode: "zero", "edge", or "reflect"

    Returns
    -------
    output_stack : ndarray
        the filtered stack of images (same size as input)
    gpu_times : tuple of float
        timing information
    """
    # create gaussian kernel on cpu
    kernel = create_gaussian_kernel_2d(sigma, kernel_size)
    kernel_size = kernel.shape[0]

    num_images, height, width = image_stack.shape

    # map padding mode to integer
    padding_modes = {"zero": 0, "edge": 1, "reflect": 2}
    if padding not in padding_modes:
        raise ValueError(f"Unknown padding mode: {padding}")
    padding_mode = padding_modes[padding]

    # compile the kernel
    kernel_file = Path("kernel_gaussian_filter_2d.cu")
    if not kernel_file.exists():
        raise FileNotFoundError(
                f"Kernel file '{kernel_file}' not found! "
                "Make sure the .cu file is in the same directory as "
                "this script."
                )
    kernel_code = kernel_file.read_text()
    start_time = time.perf_counter()

    # TODO: create a CuPy RawKernel for the padded version
    # use kernel function name "gaussian_filter_2d_padded"
    # gaussian_filter_kernel = ...

    compile_time = time.perf_counter() - start_time

    # transfer data to gpu and initialise output
    start_time = time.perf_counter()
    d_input_stack = cp.asarray(image_stack)

    # TODO: create output array with same size as input
    # d_output_stack = ...

    # TODO: transfer the flattened kernel weights to GPU
    # d_kernel_weights = ...

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
    # kernel_size = ...
    # padding_mode = ...

    # process each image in the stack
    start_time = time.perf_counter()
    for img_idx in range(num_images):
        # TODO: launch the padded kernel
        # the kernel expects these parameters in order:
        # 1. Input image (d_input_stack[img_idx])
        # 2. Output image (d_output_stack[img_idx])
        # 3. Kernel weights (d_kernel_weights)
        # 4. Height (height)
        # 5. Width (width)
        # 6. Kernel size (kernel_size)
        # 7. Padding mode (padding_mode)
        pass
    process_time = time.perf_counter() - start_time

    # transfer to CPU
    start_time = time.perf_counter()
    output_stack = d_output_stack.get()
    d2h_time = time.perf_counter() - start_time

    return output_stack, (compile_time, h2d_time, process_time, d2h_time)


def compare_performance(image_stack, sigma=1.5):
    """Compare CPU and GPU performance for Gaussian filtering."""
    print(f"\nComparing performance for {image_stack.shape[0]} images "
          f"of size {image_stack.shape[1]}x{image_stack.shape[2]}")
    print(f"Gaussian filter sigma: {sigma}")

    # time cpu version
    print("\nCPU processing (valid mode)...")
    start_time = time.perf_counter()
    cpu_result = cpu_gaussian_filter(image_stack, sigma)
    cpu_time = time.perf_counter() - start_time
    print(f"CPU time: {cpu_time:.3f} s")
    print(f"Output size: {cpu_result.shape[1]}x{cpu_result.shape[2]}")

    # time gpu version
    print("\nGPU processing (valid mode)...")
    gpu_result, gpu_times = gpu_gaussian_filter(image_stack, sigma)
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


def compare_performance_padding(image_stack, sigma=1.5, padding="edge"):
    """Compare CPU and GPU performance for Gaussian filtering with padding."""
    print(f"\nComparing performance for {image_stack.shape[0]} images "
          f"of size {image_stack.shape[1]}x{image_stack.shape[2]}")
    print(f"Gaussian filter sigma: {sigma}")
    print(f"Padding mode: {padding}")

    # time cpu version
    print("\nCPU processing (with padding)...")
    start_time = time.perf_counter()
    cpu_result = cpu_gaussian_filter_padded(image_stack, sigma,
                                           padding=padding)
    cpu_time = time.perf_counter() - start_time
    print(f"CPU time: {cpu_time:.3f} s")
    print(f"Output size: {cpu_result.shape[1]}x{cpu_result.shape[2]}")

    # time gpu version
    print("\nGPU processing (with padding)...")
    gpu_result, gpu_times = gpu_gaussian_filter_padded(image_stack, sigma,
                                                      padding=padding)
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
                                              noise_type="gaussian")
    # sigma value for gaussian filter
    sigma = 1.5

    # run performance comparison
    compare_performance(image_stack, sigma=sigma)

    # with padding
    print("="*60)
    compare_performance_padding(image_stack, sigma=sigma, padding="edge")


if __name__ == "__main__":
    sys.exit(main())
