## 3. Multidimensional grids and data

#### 1.
**a.** 
```c
__global__ void matrixMultiplyKernel1(float* A_d, float* B_d, float* C_d, int m, int n) {
    // one output row per thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n) {
        for (int col = 0; col < m; ++col) {
            float res = 0;
            for (int k = 0; k < m; ++k) {
                res += A_d[row * m + k] * B_d[col + k * m];
            }
            C_d[row * m + col] = res;
        }
    }
}
``` 
w.r.t. to the configuration, we have a grid dimension of (1, ceil(n / nthreads), 1) because we have threads in the vertical dimension only: one will output each row.

w.r.t. the kernel, each thread has an internal for corresponding to the column dimension.

**b.** 
```c
__global__ void matrixMultiplyKernel2(float* A_d, float* B_d, float* C_d, int m, int n) {
    // one output column per thread
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < m) {
        for (int row = 0; row < m; ++row) {
            float res = 0;
            for (int k = 0; k < m; ++k) {
                res += A_d[row * m + k] * B_d[col + k * m];
            }
            C_d[row * m + col] = res;
        }
    }
}
```
w.r.t. to the configuration, we have a grid dimension of (ceil(n / nthreads), 1, 1) because we have threads in the horizontal dimension only: one will output all columns.

w.r.t. the kernel, each thread has an internal for corresponding to the row dimension.

**c.** These solutions are good for when the matrix is really big and will not fit into memory. For a square matrix, both solutions are equal. When the output matrix has n > m (n = rows, m = columns), solution A is best. When m > n, solution B is best.

#### 2.

This is essentially the same as 1a but with only 1 output column.

#### 3.
**a.**
16*32

**b.**
48640

**c.**
19*5

**d.**
4500

#### 4.
**a.**
20*400 + 10 = 8010

**b.**
10*500 + 20 = 5020

#### 5.
zmn + mx + y = 5 * 400 * 500 + 400 * 10 + 20 = 1004020

