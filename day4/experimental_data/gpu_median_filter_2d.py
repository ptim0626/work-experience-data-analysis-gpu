#!/usr/bin/env python3
"""
GPU median filtering on experimental data

This script saves the output processed by GPU into an hdf5 file.
"""
import cupy as cp
import numpy as np
import h5py
from pathlib import Path
import sys
import time

from load_experimental_data import load_i14_merlin


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
    median_filter_kernel = cp.RawKernel(kernel_code,
                                        "median_filter_2d_padded")
    compile_time = time.perf_counter() - start_time

    # transfer data to gpu and initialise output
    start_time = time.perf_counter()
    d_input_stack = cp.asarray(image_stack)
    d_output_stack = cp.zeros_like(d_input_stack)
    h2d_time = time.perf_counter() - start_time

    # define thread block and grid dimensions
    threads_per_block = (16, 16)
    blocks_per_grid_x = ((width + threads_per_block[0] - 1) //
                          threads_per_block[0])
    blocks_per_grid_y = ((height + threads_per_block[1] - 1) //
                          threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # ensure correct data types
    height = cp.int32(height)
    width = cp.int32(width)
    window_size = cp.int32(window_size)
    padding_mode = cp.int32(padding_mode)

    # process each image in the stack
    start_time = time.perf_counter()
    for img_idx in range(num_images):
        # launch kernel for this image
        median_filter_kernel(blocks_per_grid,
                             threads_per_block,
                             (d_input_stack[img_idx],
                              d_output_stack[img_idx],
                              height, width, window_size,
                              padding_mode)
                             )
    process_time = time.perf_counter() - start_time

    # transfer to CPU
    start_time = time.perf_counter()
    output_stack = d_output_stack.get()
    d2h_time = time.perf_counter() - start_time

    return output_stack, (compile_time, h2d_time, process_time, d2h_time)


def main():
    # load experimental data (i14 Merlin)
    scan_file = ""

    # the output filtered file
    filtered_file = ""

    # window size used (assume maximum window size is 9)
    ws = 3

    # load the data row by row
    in_row_of = 50
    dset = None
    gpu_total_all = 0
    gpu_total_compile = 0
    gpu_total_h2d = 0
    gpu_total_process = 0
    gpu_total_d2h = 0
    with h5py.File(filtered_file, "w") as fout:
        load_data = load_i14_merlin(scan_file, in_row_of=in_row_of)
        for k, (image_stack, x_size, row_start, row_end) in enumerate(load_data):
            # GPU median filter
            gpu_result, gpu_times = gpu_median_filter_padded(image_stack, ws)

            # reshape
            filtered_4d = gpu_result.reshape(-1, x_size, *gpu_result.shape[1:])

            # record the time
            gpu_total_all += sum(gpu_times)
            gpu_total_compile += gpu_times[0]
            gpu_total_h2d += gpu_times[1]
            gpu_total_process += gpu_times[2]
            gpu_total_d2h += gpu_times[3]

            if k == 0:
                # the first chunk, initialise the dataset
                dset = fout.create_dataset("/data",
                                           shape=filtered_4d.shape,
                                           dtype=np.uint16,
                                           maxshape=(None, x_size, *gpu_result.shape[1:]),
                                           chunks=(1, 1, *gpu_result.shape[1:]),
                                           )
            else:
                # resize the dataset if it is not the first chunk
                dset.resize(in_row_of*(k+1), axis=0)

            # save the binned result into the correct place
            dset[row_start:row_end, ...] = filtered_4d

    print(f"GPU time:")
    print(f"    Total  : {gpu_total_all:.3f} s")
    print(f"    Compile: {gpu_total_compile:.3f} s")
    print(f"    H2D    : {gpu_total_h2d:.3f} s")
    print(f"    Process: {gpu_total_process:.3f} s")
    print(f"    D2H    : {gpu_total_d2h:.3f} s")


if __name__ == "__main__":
    sys.exit(main())
