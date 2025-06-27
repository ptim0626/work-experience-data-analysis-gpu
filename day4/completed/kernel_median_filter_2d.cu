extern "C" __global__
void median_filter_2d(
    const float* input,
    float* output,
    const int in_height,
    const int in_width,
    const int window_size)
{
    // calculate global thread indices for 2D processing
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // calculate half window for neighbourhood
    int half_window = window_size / 2;

    // calculate output dimensions (valid mode - smaller than input)
    int out_height = in_height - 2 * half_window;
    int out_width = in_width - 2 * half_window;

    // check if thread is within output bounds
    if (row < out_height && col < out_width) {
        // adjust indices to input image coordinates
        int in_row = row + half_window;
        int in_col = col + half_window;

        // allocate array for window values
        // maximum window size we support is 9x9 = 81 elements
        float window[81];
        int window_count = window_size * window_size;

        // collect neighbourhood values
        int idx = 0;
        for (int wr = -half_window; wr <= half_window; wr++) {
            for (int wc = -half_window; wc <= half_window; wc++) {
                int sample_row = in_row + wr;
                int sample_col = in_col + wc;

                // calculate linear index in input image
                int input_idx = sample_row * in_width + sample_col;
                window[idx] = input[input_idx];
                idx++;
            }
        }

        // bubble sort the window values
        for (int i = 0; i < window_count; i++) {
            for (int j = 0; j < window_count - i - 1; j++) {
                if (window[j] > window[j + 1]) {
                    // swap elements
                    float temp = window[j];
                    window[j] = window[j + 1];
                    window[j + 1] = temp;
                }
            }
        }

        // find median (middle element)
        int middle_index = window_count / 2;
        float median_value = window[middle_index];

        // write to output
        int output_idx = row * out_width + col;
        output[output_idx] = median_value;
    }
}

extern "C" __global__
void median_filter_2d_padded(
    const float* input,
    float* output,
    const int height,
    const int width,
    const int window_size,
    const int padding_mode)
{
    // calculate global thread indices for 2D processing
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // check if thread is within image bounds
    if (row < height && col < width) {
        // calculate half window for neighbourhood
        int half_window = window_size / 2;

        // allocate array for window values
        float window[81];
        int window_count = window_size * window_size;

        // collect neighbourhood values with padding handling
        int idx = 0;
        for (int wr = -half_window; wr <= half_window; wr++) {
            for (int wc = -half_window; wc <= half_window; wc++) {
                int sample_row = row + wr;
                int sample_col = col + wc;

                // handle padding based on mode
                if (padding_mode == 0) {
                    // zero padding
                    if (sample_row < 0 || sample_row >= height ||
                        sample_col < 0 || sample_col >= width) {
                        window[idx] = 0.0f;
                    } else {
                        int input_idx = sample_row * width + sample_col;
                        window[idx] = input[input_idx];
                    }
                } else if (padding_mode == 1) {
                    // edge padding - clamp to edge values
                    sample_row = max(0, min(height - 1, sample_row));
                    sample_col = max(0, min(width - 1, sample_col));
                    int input_idx = sample_row * width + sample_col;
                    window[idx] = input[input_idx];
                } else {
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
                    window[idx] = input[input_idx];
                }
                idx++;
            }
        }

        // bubble sort the window values
        for (int i = 0; i < window_count; i++) {
            for (int j = 0; j < window_count - i - 1; j++) {
                if (window[j] > window[j + 1]) {
                    // swap elements
                    float temp = window[j];
                    window[j] = window[j + 1];
                    window[j + 1] = temp;
                }
            }
        }

        // find median (middle element)
        int middle_index = window_count / 2;
        float median_value = window[middle_index];

        // write to output
        int output_idx = row * width + col;
        output[output_idx] = median_value;
    }
}
