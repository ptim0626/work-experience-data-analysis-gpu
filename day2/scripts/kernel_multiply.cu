/*
GPU Kernel: Multiply Each Element by Two
========================================
This kernel demonstrates a different mathematical operation.
Instead of addition, we perform multiplication.
*/

extern "C" __global__
void multiply(float* data, int size) {
    /*
    This kernel multiplies each array element by 2.
    The structure remains identical - only the operation changes.
    */

    // calculate global thread index
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    /*
    Remember the theatre analogy:
    - blockDim.x  = 256 seats per row
    - blockIdx.x  = your row number
    - threadIdx.x = your seat in that row

    Each thread gets a unique idx value to process
    exactly one array element.
    */

    // boundary check - essential for safety
    if (idx < size) {
        data[idx] = data[idx] * 2.0f;
        /*
        GPU flexibility:
        - same parallel structure
        - same memory access pattern
        - different computation
        - no change to thread organisation

        You could change this to:
        - data[idx] * 3.0f    (triple)
        - data[idx] / 2.0f    (halve)
        - data[idx] * data[idx] (square)
        */
    }
}
