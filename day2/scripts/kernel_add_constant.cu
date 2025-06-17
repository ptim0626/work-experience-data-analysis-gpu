/*
GPU Kernel: Add Constant to Array
==================================
This CUDA C kernel adds a constant value to each element of an array.
Each thread processes exactly one element, enabling massive parallelism.

Remember: On the GPU, thousands of threads run this same code simultaneously.
(SPMD)
*/

extern "C" __global__
void add_constant(float* data, int size) {
    /*
    This is CUDA C code that runs on the GPU.
    Let's break down the special keywords:

    extern "C"  - tells the compiler to use C naming conventions
                  (makes it easier to call from Python)
    __global__  - marks this function as:
                  > Callable from CPU (host)
                  > Executable on GPU (device)

    Parameters explained:
    float* data - a pointer to our array in GPU memory
                  (the * means "pointer to" - it's the address where
                   our array lives in GPU memory)
    int size    - total number of elements in the array
    */

    // calculate which element THIS particular thread should process
    // every thread runs this same code but gets different index values
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    /*
    Understanding the index calculation - THE MOST IMPORTANT PART!
    =============================================================

    Think of threads organised like a theatre:
    - blockDim.x  = seats per row (threads per block)
    - blockIdx.x  = which row you're in (block number)
    - threadIdx.x = your seat number in that row

    Example with actual values from our programme:
    - blockDim.x = 256 (256 seats per row)
    - blockIdx.x = 2 (you're in row 2, counting from 0)
    - threadIdx.x = 47 (you're in seat 47 of your row)

    Your unique position: 256 * 2 + 47 = 559
    So THIS thread processes element 559 of the array.

    Visual representation with smaller numbers for clarity:
    If we had blockDim.x = 4 (only 4 seats per row):

    Row 0: [0] [1] [2] [3]      (threads 0-3)
    Row 1: [4] [5] [6] [7]      (threads 4-7)
    Row 2: [8] [9] [10] [11]    (threads 8-11)
    Row 3: [12] [13] [14] [15]  (threads 12-15)

    Thread in row 2, seat 1 handles element: 4 * 2 + 1 = 9
    */

    // SAFETY CHECK: ensure this thread has valid work to do
    if (idx < size) {
        // THE ACTUAL WORK: add 10 to our assigned element
        // each thread only modifies ONE element
        data[idx] = data[idx] + 10.0f;
        /*
        Note: The 'f' in 10.0f tells the compiler this is a
        32-bit float, not a 64-bit double. This matches our
        array type and is faster on GPUs.

        To change the value being added, simply change 10.0f
        to any other number, like 25.0f or -5.0f
        */
    }
    /*
    Why the if statement?
    ====================
    GPUs launch threads in groups (warps of 32, blocks of 32-1024).
    If we have 1000 elements but launch 1024 threads (4 blocks of 256),
    threads 1000-1023 would try to access memory beyond our array!
    This check prevents crashes and corrupted data.
    */
}
