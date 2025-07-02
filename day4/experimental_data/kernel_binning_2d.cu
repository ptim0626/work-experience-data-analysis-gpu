extern "C" __global__
void pixel_binning_2d(
    const float* input,
    float* output,
    const int in_height,
    const int in_width,
    const int bin_size)
{
    // calculate global thread indices for 2D processing
    int bin_col = blockIdx.x * blockDim.x + threadIdx.x;
    int bin_row = blockIdx.y * blockDim.y + threadIdx.y;

    // calculate output dimensions
    int out_height = (in_height + bin_size - 1) / bin_size;
    int out_width = (in_width + bin_size - 1) / bin_size;

    // check if thread is within output bounds
    if (bin_row < out_height && bin_col < out_width) {
        // initialise sum for this bin
        float bin_sum = 0.0f;

        // sum all pixels belonging to this bin
        for (int dy = 0; dy < bin_size; dy++) {
            for (int dx = 0; dx < bin_size; dx++) {
                // calculate source pixel position
                int src_row = bin_row * bin_size + dy;
                int src_col = bin_col * bin_size + dx;

                // boundary check - critical to avoid out-of-bounds access
                if (src_row < in_height && src_col < in_width) {
                    // calculate linear index in input image
                    int input_idx = src_row * in_width + src_col;
                    bin_sum += input[input_idx];
                }
            }
        }

        // write sum to output
        int output_idx = bin_row * out_width + bin_col;
        output[output_idx] = bin_sum;
    }
}
