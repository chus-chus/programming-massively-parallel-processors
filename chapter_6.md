## 6. Performance considerations

#### 1.
I don't have an nvidia gpu right now to test this kernel!!!!!
```c
// C = A*B, where A is stored in a row-major layout and B in a column-major layout
// A and B are squared matrices
// BLOCK_WIDTH = TILE_WIDTH
// matrix width is multiple of TILE_WIDTH

#define TILE_WIDTH 16
__global__ 
void matmul_corner_turning(float* A, float* B, float* C, int width) {

    __shared__ float A_s[TILE_WIDTH][TILE_WIDTH];
    __shared__ float B_s[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int tx = threadIdx.x;
    int by = blockIdx.y; int ty = threadIdx.y;

    // row and column index of output element
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float outputVal = 0.f;
    for (int ph = 0; ph < width/TILE_WIDTH; ++ph) {
        // Accesses to A coalesce naturally
        A_s[ty][tx] = A[row * width + ph * TILE_WIDTH + col];
        /*  Accesses to B need to be transposed for memory access coalescing: corner turning
            In this style, consecutive threads in the x dimension will access consecutive elements in the y dimension,
            so global memory accesses within warps will be coalesced (not completely if TILE_WIDTH != 32).
            - ph * TILE_WIDTH * width -> locate "phase tile" in the y dimension
            - tx * width -> locate row on "phase tile" depending on the threadIdx.x position
            - row -> move to the appropiate x position of the "phase tile" depending on the y axis of the thread */
        B_s[ty][tx] = B[ph * TILE_WIDTH * width + tx * width + row];
        __syncthreads();
        
        for (int k = 0; k < TILE_WIDTH; ++k) {
            outputVal += A_s[ty][k] * B_s[k][tx];
        }
        __syncthreads();
    }
    C[row * width + col] = outputVal;
}
```

#### 2.
I gave a hint in the previous answer. Global memory accesses when BLOCK_WIDTH = 32 will all be coalesced. Remember that warp size is 32? All threads in a warp are consecutive in the x dimension, so if a warp corresponds to a whole row of the matrices, all threads in the warp will access consecutive elements in memory. So, all accesses will be coalesced.

#### 3.
**a.** Coalesced: all consecutive threads access consecutive elements in memory.  
**b.** Not applicable: SRAM does not require coalescing.  
**c.** Coalesced: at any iteration, consecutive threads access consecutive elements in memory (only i changes between threads in a block).  
**d.** Not coalesced: consecutive threads in a block will access elements separated by 4 elements in memory.  
**e.** Not applicable because of the same reason as **b.**  
**f.** Again, not applicable.  
**g.** Coalesced.  
**h.** Not applicable.  
**i.** Not coalesced: consecutive threads in a block will access elements separated by 8 elements in memory.  

#### 4.  
Note that we assume each float is a 4-byte value.  
**a.** Each thread (there are w^2, where width=w), performs w products, w additions and 2w global memory accesses. This gives a flop to global memory access ratio of 2w / 2w = 1 flop per access and 0.25 flops per byte.  
**b.** Each thread (again, there are w^2), performs the same number of adds and multiplies (2w), but now only performs (w / tile_width) * 2 global memory accesses. So, the flop to gma ration is 2w / (2w / tile_width) = tile_width . So, 32 flops per access and 8 flops per byte.  
**c.** This time, we have a smaller number of threads to perform the same number of adds and multiplies. In particular, because each thread now is responsible for computing COARSE_FACTOR (c) output elements, each one performs 2wc floating operations. Now, each thread accesses global memory (w / tile_width) + (w / tile_width) * c = (1 + c) * (w / tile_width) times. So, the flop to gma ratio is 2wc / (1 + c) * (w / tile_width). When c = 4 and tile_width = 32, this is 8w/(5*w/32) = 32/5 * 8 = 51,2 flops per access and 12,8 flops per byte.