## 7. Convolution

#### 1.
51 (I assume they mean the element y[0], not P[0]).

#### 2.
8, 21, 13, 20, 7

#### 3.
**a.** Identity kernel  
**b.** Shift left  
**c.** Shift right  
**d.** Change detection (vertical in the case of 2D input)  
**e.** Avg.

#### 4.
**a.** $2r$, where $r$ is the radius of the filter. $\lfloor M/2\rfloor$.

**b.** $NM$

**c.**
$$
\begin{align}
N\times M - 2(r + (r-1) + (r-2) + ... + 1) &= \nonumber \\
NM - 2\sum_{i=0}^{r-1}{(r-i)} &= \nonumber \\
NM - 2\frac{r(r+1)}{2} &= \nonumber \\
NM - r(r+1) \nonumber \\
\end{align}
$$

#### 5.

**a.** $4Mr + 4r^2$

**b.** $N^2 M^2$

**c.** This is more complicated than before, and I don't have time to latex! 

Let's first define the filter radius as:

r = ⌊m1/2⌋

this is what essentially overlaps with ghost cells. We compute the total number of mults as the sum of three components: 1) when the input cell multiplies all filter cells, 2) when the filter is in a corner case (ghost cells on two sides), and 3) when the filter is on a side (ghost cells on one side).

1) number of elements with full multiply = (n-2r)(n-2r)(m*m)
2) all edges: 4*[(n-2p) * sum(from i=1 to p)(m*i)] 
3) 4*[sum(from i=1 to p)(sum(from j=1 to p)(i*j))]

#### 6.

We define the radiuses:

rr = ⌊m1/2⌋ (radius rows)  
rc = ⌊m2/2⌋ (radius columns)

**a.** $2(N_1r_2 + N_2 r_1) + (4r_1r_2)$. The first term corresponds to sides, and the second to corners.

**b.** $N_1N_2M_1M_2$

**c.** We apply the same concepts as in 5c, but now we have the vertical and horizontal dimensions to consider.

1) number of elements with full multiply = (n1-2* pr)(n2-2* pc)(m1*m2)
2) corners: 4*[sum(from i=1 to rr)(sum(from j=1 to rc)(i*j))]
3) left & right: 2*[(n1-2* rr) * sum(from i=1 to rc)(i* m2)]  
  top and bottom: 2*[(n2-2* rc) * sum(from i=1 to rr)(m1*i)]

#### 7.

```c
__global__
void convolution_3D_basic_kernel (float *N, float *F, float *P, int r, int width, int height, int depth){ {
    int outCol = blockIdx.x*blockDim.x + threadIdx.x;
    int outRow = blockIdx.y*blockDim.y + threadIdx.y;
    int outDepth = blockIdx.z*blockDim.z + threadIdx.z;
    float Pvalue = 0.0f;
    for (int fRow = 0; fRow < 2*r+1; fRow++) {
        for (int fCol = 0; fCol < 2*r+1; fCol++) {
            for (int fDepth = 0; fDepth < 2*r+1; fDepth++) {
                int inRow = outRow - r + fRow;
                int inCol = outCol - r + fCol;
                int inDepth = outDepth - r + fDepth;
                if (inRow >= 0 && inRow < height && inCol >= 0 && 
                    inCol < width && inDepth >= 0 && inDepth < depth) {
                    Pvalue += F[fRow][fCol][fDepth] * N[inDepth*width*height + inRow*width + inCol];
                }
            }
    P[outDepth*width*height + outRow*width + outCol] = Pvalue;
}
```

#### 8.

Identical to the previous one, but F not as a parameter.

#### 9.

```c
#define IN_TILE_DIM 32
#define OUT_TILE_DIM ((IN TILE DIM) - 2*(FILTER RADIUS))
__constant__ float F_c[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];
__global__ void convolution_tiled_3D_const_mem_kernel (float *N, float *P, int width, int height, int depth) {
    int col = blockIdx.x*OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
    int row = blockIdx.y*OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;
    int out_depth = blockIdx.z*OUT_TILE_DIM + threadIdx.z - FILTER_RADIUS;
    //loading input tile
    __shared__ float N_s[IN_TILE DIM][IN_ TILE_DIM]:
    if (row>=0 && row<height && col>=0 && col<width) {
        N_s[threadIdx.y][threadIdx.x][threadIdx.z] = 
                N[out_depth*width*height + row*width + col];
    } else {
        N_s[threadIdx.y][threadIdx.x][threadIdx.z] = 0.0f;
    }
    __syncthreads();
    // Calculating output elements
    int tileCol = threadIdx.x - FILTER_RADIUS;
    int tileRow = threadIdx.y - FILTER_RADIUS;
    int tileDepth = threadIdx.z - FILTER_RADIUS;
    // turning off the threads at the edges of the block
    if (col >= 0 && col < width && row >=0 && row < height && 
        out_depth >= 0 && out_depth < depth) {
        if (tileCol >= 0 && tileCol < OUT_TILE_DIM && tileRow>=0
        && tileRow<OUT_IILE_DIM && tileDepth >= 0 && tileDepth < OUT_TILE_DIM) {
            float Pvalue = 0.0f;
            for (int fRow = 0; fRow < 2*FILTER_RADIUS+1; fRow++) {
                for (int fCol = 0; fCol < 2*FILTER_RADIUS+1; {Col++) {
                    for (int fDepth = 0; fDepth < 2*FILTER_RADIUS+1; fDepth++) {
                        Pvalue += F_c[fRow][fCol][fDepth] * 
                                  N_s[tileDepth+fDepth][tileRow+fRow][tileCol+fCol];
                    }
                }
            }
            P[outDepth*width*height + outRow*width + outCol] = Pvalue;
        }
    }
}
```
