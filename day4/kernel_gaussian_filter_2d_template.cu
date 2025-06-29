extern "C" __global__
void gaussian_filter_2d(
    const float* input,
    float* output,
    const float* kernel_weights,
    const int in_height,
    const int in_width,
    const int kernel_size)
{
    // TODO: calculate global thread indices for 2D processing
    // Hint: use blockIdx, blockDim, and threadIdx
    // Remember: col uses .x components, row uses .y components
    // int col = ...
    // int row = ...

    // calculate half kernel size for neighbourhood
    int half_kernel = kernel_size / 2;

    // calculate output dimensions (valid mode - smaller than input)
    int out_height = in_height - 2 * half_kernel;
    int out_width = in_width - 2 * half_kernel;

    // TODO: check if thread is within output bounds
    // only process if both row and col are valid
    if (false) {  // Fix this condition
        // adjust indices to input image coordinates
        int in_row = row + half_kernel;
        int in_col = col + half_kernel;

        // TODO: initialise variable to accumulate weighted sum
        // float weighted_sum = ...

        // TODO: apply convolution with gaussian kernel
        // use nested loops to iterate over the kernel
        // for each kernel position:
        //   1. calculate position in input image
        //   2. calculate linear indices for input and kernel
        //   3. multiply pixel value by kernel weight
        //   4. add to weighted_sum
        for (int kr = 0; kr < kernel_size; kr++) {
            for (int kc = 0; kc < kernel_size; kc++) {
                // TODO: calculate position in input image
                // int sample_row = ...
                // int sample_col = ...

                // TODO: calculate linear indices
                // int input_idx = ...
                // int kernel_idx = ...

                // TODO: multiply and accumulate
                // weighted_sum += ...
            }
        }

        // TODO: write result to output at correct location
        // calculate the linear index for the output array
        // int output_idx = ...
        // output[output_idx] = ...
    }
}

extern "C" __global__
void gaussian_filter_2d_padded(
    const float* input,
    float* output,
    const float* kernel_weights,
    const int height,
    const int width,
    const int kernel_size,
    const int padding_mode)
{
    // TODO: calculate global thread indices
    // int col = ...
    // int row = ...

    // TODO: check if thread is within image bounds
    if (false) {  // Fix this condition
        // calculate half kernel size for neighbourhood
        int half_kernel = kernel_size / 2;

        // TODO: initialise variables for weighted sum and weight sum
        // we need weight_sum for normalisation at edges
        // float weighted_sum = ...
        // float weight_sum = ...

        // apply convolution with gaussian kernel
        for (int kr = 0; kr < kernel_size; kr++) {
            for (int kc = 0; kc < kernel_size; kc++) {
                // TODO: calculate position in input image
                // int sample_row = ...
                // int sample_col = ...

                // TODO: get kernel weight for this position
                // int kernel_idx = ...
                // float kernel_weight = ...

                // handle padding based on mode
                float pixel_value = 0.0f;
                bool valid_pixel = false;

                if (padding_mode == 0) {
                    // TODO: implement zero padding
                    // check if sample position is within bounds
                    // if yes: get pixel value and set valid_pixel = true
                    // if no: use 0.0f and set valid_pixel = true

                } else if (padding_mode == 1) {
                    // TODO: implement edge padding
                    // clip indices to image boundaries using max() and min()
                    // sample_row = ...
                    // sample_col = ...
                    // get pixel value and set valid_pixel = true

                } else if (padding_mode == 2) {
                    // TODO: implement reflect padding
                    // reflect indices at boundaries
                    // if (sample_row < 0) ...
                    // else if (sample_row >= height) ...
                    // (same for sample_col)
                    // then clip to ensure within bounds
                    // get pixel value and set valid_pixel = true

                }

                // TODO: accumulate weighted sum if pixel is valid
                // if (valid_pixel) {
                //     weighted_sum += ...
                //     weight_sum += ...
                // }
            }
        }

        // TODO: normalise by actual weight sum to handle edge cases
        // float result = ...
        // if (weight_sum > 0.0f) {
        //     result = ...
        // }

        // TODO: write result to output
        // int output_idx = ...
        // output[output_idx] = ...
    }
}
