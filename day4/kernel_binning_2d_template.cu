extern "C" __global__
void pixel_binning_2d(
    const float* input,
    float* output,
    const int in_height,
    const int in_width,
    const int bin_size)
{
    // TODO: calculate global thread indices for 2D processing
    // Remember: col uses .x components, row uses .y components
    // int bin_col = ...
    // int bin_row = ...

    // calculate output dimensions
    int out_height = (in_height + bin_size - 1) / bin_size;
    int out_width = (in_width + bin_size - 1) / bin_size;

    // TODO: check if thread is within output bounds
    // only process if both bin_row and bin_col are valid
    if (false) {  // Fix this condition
        // TODO: initialise sum for this bin
        // float bin_sum = ...

        // TODO: sum all pixels belonging to this bin
        // use nested loops to iterate over the bin_size x bin_size region
        // for (int dy = 0; dy < ...; dy++) {
        //     for (int dx = 0; dx < ...; dx++) {
                // TODO: calculate source pixel position
                // int src_row = ...
                // int src_col = ...

                // TODO: add boundary check - critical to avoid out-of-bounds access
                // only add pixel value if within input image bounds
                // if (...) {
                    // TODO: calculate linear index in input image
                    // int input_idx = ...

                    // TODO: add pixel value to bin sum
                    // bin_sum += ...
                // }
        //     }
        // }

        // TODO: write sum to output at correct location
        // calculate the linear index for the output array
        // int output_idx = ...
        // output[output_idx] = ...
    }
}
