extern "C" __global__
void brightness_adjust_2d(
    const float* input,
    float* output,
    const int height,
    const int width,
    const float brightness)
{
    // calculate global thread indices for 2D processing
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // check if thread is within image bounds
    if (row < height && col < width) {
        // calculate linear index
        int idx = row * width + col;

        // read input value and add brightness
        float value = input[idx] + brightness;

        // write to output
        output[idx] = value;
    }
}
