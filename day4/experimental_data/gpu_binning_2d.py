#!/usr/bin/env python3
"""
GPU pixel binning on experimental data

This script saves the output processed by GPU into an hdf5 file.
"""
import cupy as cp
import numpy as np
import h5py
from pathlib import Path
import sys
import time

from load_experimental_data import load_i14_excalibur


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


def main():
    # load experimental data (i14 Excalibur)
    scan_file = ""

    # the output binned file
    binned_file = ""
    bin_size = 4

    # load the data chunk by chunk
    sum_frame = 0
    in_chunk_of = 50
    dset = None
    gpu_total_all = 0
    gpu_total_compile = 0
    gpu_total_h2d = 0
    gpu_total_process = 0
    gpu_total_d2h = 0
    with h5py.File(binned_file, "w") as fout:
        load_data = load_i14_excalibur(scan_file, in_chunk_of=in_chunk_of)
        for k, image_stack in enumerate(load_data):
            # GPU binning
            gpu_result, gpu_times = gpu_bin_pixels(image_stack, bin_size)

            # remove hot pixel manually
            gpu_result[gpu_result > 1000] = 0

            # record the time
            gpu_total_all += sum(gpu_times)
            gpu_total_compile += gpu_times[0]
            gpu_total_h2d += gpu_times[1]
            gpu_total_process += gpu_times[2]
            gpu_total_d2h += gpu_times[3]

            if k == 0:
                # the first chunk, initialise the dataset
                dset = fout.create_dataset("/data",
                                           shape=gpu_result.shape,
                                           dtype=gpu_result.dtype,
                                           maxshape=(None, *gpu_result.shape[1:]),
                                           chunks=(1, *gpu_result.shape[1:]),
                                           )
            else:
                # resize the dataset if it is not the first chunk
                dset.resize(in_chunk_of*(k+1), axis=0)

            # save the binned result into the correct place
            dset[in_chunk_of*k:in_chunk_of*(k+1), ...] = gpu_result

            # integrate the frames
            sum_frame += gpu_result.sum(axis=0)

        # save the integrated frame
        fout["/sum"] = sum_frame

    print(f"GPU time:")
    print(f"    Total  : {gpu_total_all:.3f} s")
    print(f"    Compile: {gpu_total_compile:.3f} s")
    print(f"    H2D    : {gpu_total_h2d:.3f} s")
    print(f"    Process: {gpu_total_process:.3f} s")
    print(f"    D2H    : {gpu_total_d2h:.3f} s")


if __name__ == "__main__":
    sys.exit(main())
