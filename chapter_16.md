## 7. Deep learning

#### 1.

It's very similar to the convolution kernel in fig 16.15, but now the number of output elements across each dim is dim / K. Each thread will compute one output element, and so the the mapping of thread h and w, and input h and w, changes slightly.

```c
M = 4;
N = 3;
K = 2;
W_out = W_in / K;
H_out = H_in / K;
W_grid = W_out / tile_width;
H_grid = H_out / tile_width;
T = W_grid * H_grid;
dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
dim3 gridDim(M, T, N);

__global__ void PoolingLayerForward_Kernel (int W_grid, int K, float* Y) {
    int m = blockIdx.x;
    int h = (blockIdx.y / W_grid) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.y % W_grid) * TILE_WIDTH + threadIdx.x;
    int n = blockIdx.z;
    float acc = 0.;
    for (int p = 0; p < K; p++) {
        for (int q = 0; q < K; qt+) {
            acc += Y[n, m, K*h + p, K*w + q];
        }
    }
    Y[n, m, h, w] = acc / (K*K);
}
```

#### 2.

**a.** In this case, elements adjacent in memory are those in the same w and h positions, and consecutive input channels. That is, elements e1, e2 and e3, which are memory adjacent, correspond (for example), to width w=1, height=1 and channels 1, 2, and 3. This is in contrast to the layout we have been using, [N, C, H, W], where memory-adjacent elements are those in the same input channel, with same height index, and consecutive width index. The new approach could indeed reduce the memory bw utilization, for example when we do a 3D convolution and the number of channels exceeds the width of the 3D filter that we are using. In this case, memory accesses for elements across channels can be coalesced. The convolution kernel in figure 16.15 will not see this improvement because it performs a 2D convolution across a single channel, so all memory accesses will be physically apart.  
**b.** If we use a [C, H, W, N] layout, memory-adjacent elements are those with same channel, width and height indices, and belong to consecutive input images. In this case, we could see memory coalescing and optimize shared memory and global memory access if we designed the kernel to work across images.

#### 3.

First, the backprop kernel for the dE/dx computation. Each thread computes one element of dE/dx for a single input channel. We organise the grid similarly as to the convolution kernel: N, T, C. T is the number of tiles that each block of threads has.

```c
__global__ void ConvBackpropXGrad_Kernel (int M, int C, int H, int W, int K, float* dE_dy, float* W, float* dE_dx) {
    int n = blockIdx.z;
    int c = blockIdx.x;
    // same as before
    int h = (blockIdx.y / W_grid) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.y % W_grid) * TILE_WIDTH + threadIdx.x;

    grad = 0.f;
    // backwards convolution accross all output channels
    for (int m = 0; m < M; ++m) {
        for (int p = 0; p < K, ++p) {
            for (int q = 0; q < K, ++q) {
                if (h-p >= 0 && w-p >= 0 h-p < H && w-q < W) {
                    grad += dE_dy[n, m, h - p, w - q] * W[n, m, K - p, K - q];
                }
            }
        }
    }
    dE_dx[n, c, h, w] = grad;
}
```

Second, the backprop kernel for dE/dw. Here we face the decision of what a block computes. If we make each block compute the gradients for a single filter, filters need a minimum of 32 elements (a warp is 32 threads). This usually does not happen, so we could also make each block compute the gradients of a whole filter bank. Now, if each thread computes one gradient element, we will most likely have at least 32 threads in each block, depending on the sizes of the filters and number of input channels. With this computing layout, the grid dimensions would be N, 1, M, where N is the number of inputs and M is the number of output channels (each block is responsible for a filter bank, and there are M). Each block would have dimensions C, K, K. That is, C * K * K threads.

```c
__global__ void ConvBackpropWGrad_Kernel (int M, int C, int H, int W, int K, float* dE_dy, float* W, float* dE_dx) {
    // block indices: locate filter bank
    int n = blockIdx.z;
    int m = blockIdx.x;
    
    // thread indices: locate element in filter bank
    int c = threadIdx.z;
    int p = threadIdx.y;
    int q = threadIdx.x;

    int H_out = H - K + 1;
    int W_out = W - K + 1;

    grad = 0.f;
    for (int h = 0; h < H_out, ++h) {
        for (int w = 0; w < W_out, ++w) {
            grad += dE_dy[n, m, h, w] * X[n, c, h + p, w + q];
        }
    }
    dE_dw[n, m, c, p, q] = grad;
}
```

I have not optimised for memory access, that's for sure :).

#### 4.

Each thread is responsible for loading a 2D K x K patch of elements of X. This 2D structure makes it so that adjacent threads access elements that are probably not contiguous in memory. This results in a non-coalesced memory access pattern.