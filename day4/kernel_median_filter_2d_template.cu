extern "C" __global__
void median_filter_2d(
    const float* input,
    float* output,
    const int in_height,
    const int in_width,
    const int window_size)
{
    // TODO: calculate global thread indices for 2D processing
    // Hint: use blockIdx, blockDim, and threadIdx
    // Remember: col uses .x components, row uses .y components
    // int col = ...
    // int row = ...

    // calculate half window for neighbourhood
    int half_window = window_size / 2;

    // calculate output dimensions (valid mode - smaller than input)
    int out_height = in_height - 2 * half_window;
    int out_width = in_width - 2 * half_window;

    // TODO: check if thread is within output bounds
    // only process if both row and col are valid
    if (false) {  // Fix this condition
        // adjust indices to input image coordinates
        int in_row = row + half_window;
        int in_col = col + half_window;

        // allocate array for window values
        // maximum window size we support is 9x9 = 81 elements
        float window[81];
        int window_count = window_size * window_size;

        // TODO: collect neighbourhood values
        // use nested loops from -half_window to +half_window
        // store values in the window array
        int idx = 0;
        // Add nested loops here


        // TODO: implement bubble sort on the window array
        // this should match the CPU version exactly
        // remember to swap elements when needed


        // TODO: find median (middle element)
        // calculate the middle index and get the value
        // float median_value = ...

        // TODO: write to output at correct location
        // Calculate the linear index for the output array
        // int output_idx = ...
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
    // TODO: calculate global thread indices
    // int col = ...
    // int row = ...

    // TODO: check if thread is within image bounds
    if (false) {  // Fix this condition
        // calculate half window for neighbourhood
        int half_window = window_size / 2;

        // allocate array for window values
        float window[81];
        int window_count = window_size * window_size;

        // TODO: collect neighbourhood values with padding handling
        // You'll need to handle three padding modes:
        // 0: zero padding
        // 1: edge padding
        // 2: reflect padding
        int idx = 0;
        for (int wr = -half_window; wr <= half_window; wr++) {
            for (int wc = -half_window; wc <= half_window; wc++) {
                int sample_row = row + wr;
                int sample_col = col + wc;

                // TODO: handle padding based on mode
                if (padding_mode == 0) {
                    // zero padding
                    // check bounds and use 0.0f for out-of-bounds

                } else if (padding_mode == 1) {
                    // edge padding - clamp to edge values
                    // use max() and min() to clamp indices

                } else {
                    // reflect padding
                    // implement reflection at boundaries

                }
                idx++;
            }
        }

        // TODO: sort and find median (same as basic version)


        // TODO: write result to output

    }
}
