extern "C" __global__
void gaussian_filter_2d(
    const float* input,
    float* output,
    const float* kernel_weights,
    const int in_height,
    const int in_width,
    const int kernel_size)
{
    // calculate global thread indices for 2D processing
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // calculate half kernel size for neighbourhood
    int half_kernel = kernel_size / 2;

    // calculate output dimensions (valid mode - smaller than input)
    int out_height = in_height - 2 * half_kernel;
    int out_width = in_width - 2 * half_kernel;

    // check if thread is within output bounds
    if (row < out_height && col < out_width) {
        // adjust indices to input image coordinates
        int in_row = row + half_kernel;
        int in_col = col + half_kernel;

        // accumulate weighted sum
        float weighted_sum = 0.0f;

        // apply convolution with gaussian kernel
        for (int kr = 0; kr < kernel_size; kr++) {
            for (int kc = 0; kc < kernel_size; kc++) {
                // calculate position in input image
                int sample_row = in_row + kr - half_kernel;
                int sample_col = in_col + kc - half_kernel;

                // calculate linear indices
                int input_idx = sample_row * in_width + sample_col;
                int kernel_idx = kr * kernel_size + kc;

                // multiply and accumulate
                weighted_sum += input[input_idx] * kernel_weights[kernel_idx];
            }
        }

        // write to output
        int output_idx = row * out_width + col;
        output[output_idx] = weighted_sum;
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
    // calculate global thread indices for 2D processing
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // check if thread is within image bounds
    if (row < height && col < width) {
        // calculate half kernel size for neighbourhood
        int half_kernel = kernel_size / 2;

        // accumulate weighted sum and weight sum for normalisation
        float weighted_sum = 0.0f;
        float weight_sum = 0.0f;

        // apply convolution with gaussian kernel
        for (int kr = 0; kr < kernel_size; kr++) {
            for (int kc = 0; kc < kernel_size; kc++) {
                // calculate position in input image
                int sample_row = row + kr - half_kernel;
                int sample_col = col + kc - half_kernel;

                // get kernel weight
                int kernel_idx = kr * kernel_size + kc;
                float kernel_weight = kernel_weights[kernel_idx];

                // handle padding based on mode
                float pixel_value = 0.0f;
                bool valid_pixel = false;

                if (padding_mode == 0) {
                    // zero padding
                    if (sample_row >= 0 && sample_row < height &&
                        sample_col >= 0 && sample_col < width) {
                        int input_idx = sample_row * width + sample_col;
                        pixel_value = input[input_idx];
                        valid_pixel = true;
                    } else {
                        pixel_value = 0.0f;
                        valid_pixel = true;
                    }
                } else if (padding_mode == 1) {
                    // edge padding - clamp to edge values
                    sample_row = max(0, min(height - 1, sample_row));
                    sample_col = max(0, min(width - 1, sample_col));
                    int input_idx = sample_row * width + sample_col;
                    pixel_value = input[input_idx];
                    valid_pixel = true;
                } else if (padding_mode == 2) {
                    // reflect padding
                    if (sample_row < 0) {
                        sample_row = -sample_row;
                    } else if (sample_row >= height) {
                        sample_row = 2 * height - sample_row - 2;
                    }

                    if (sample_col < 0) {
                        sample_col = -sample_col;
                    } else if (sample_col >= width) {
                        sample_col = 2 * width - sample_col - 2;
                    }

                    // ensure within bounds after reflection
                    sample_row = max(0, min(height - 1, sample_row));
                    sample_col = max(0, min(width - 1, sample_col));

                    int input_idx = sample_row * width + sample_col;
                    pixel_value = input[input_idx];
                    valid_pixel = true;
                }

                // accumulate if valid
                if (valid_pixel) {
                    weighted_sum += pixel_value * kernel_weight;
                    weight_sum += kernel_weight;
                }
            }
        }

        // normalise by actual weight sum to handle edge cases
        float result = 0.0f;
        if (weight_sum > 0.0f) {
            result = weighted_sum / weight_sum;
        }

        // write to output
        int output_idx = row * width + col;
        output[output_idx] = result;
    }
}
