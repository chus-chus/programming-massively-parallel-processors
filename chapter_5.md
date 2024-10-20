## 5. Memory architecture and data locality

### 1.
No, because elements are added one on one, with no commonality whatsoever.

### 2. 
Can't draw now, but the idea is that with a 4x4 block you would have 2 phases, and each phase loads 16 elements from global memory, which you reuse among the threads in the block. With a 2x2 block, you would have 4 phases, each one loading 4 elements from global memory, which you reuse among the threads in the block. Thus, a smaller block will require more accesses to global memory, because more threads will have to load the same elements.

### 3.
If we forget the first one, invalid elements may be accessed when computing the partial matmul. If one forgets the second one, a thread further down the execution stream might overwrite a value in shared memory that a thread performing the partial matmul computation might need, thus corrputing the result. 

### 4.
The whole point of shared memory is that it’s shared - and fast. If one were to use only registers, all of the threads would have to load their private, corresponding elements of the input matrices M and N, which are repeatedly access from multiple threads. This access to global memory not only is slow, but is also repeated in this case. We want shared memory so that elements from global memory are only accessed once (per block)!

### 5.
It’s a 32x reduction: elements that would be accessed by all 32 threads separately are only loaded from global memory once. Thus, its 1/32 of the original. 

### 6.
512000 versions.

### 7.
1 for each block, so 1000.

### 8.
a: N times  
b: N/T times

### 9.
Arithmetic intensity of the kernel is 36 / (7*(32/8)) = 36/28 =  1.28 ops per byte

a: for each 100GB that are loaded, the kernel would do 128GFLOPS. Peak is 200GFLOPS, so it’s memory bound.

b: for each 250GBs loaded, the kernel would do 320GFLOPS. Peak is 300GFLOPS, so it’s compute bound.

### 10.

#### a. Out of the possible range of values for BLOCK_SIZE, for what values of BLOCK_SIZE will this kernel function execute correctly on the device?

BLOCK_SIZE = 1. With bigger block size, the matrix in shared memory will be accessed by multiple threads, and because the code does not have any synchronization mechanism, a race condition may occur. For example, assigning to A_elements an element from blockA that was not yet written to, or reading from blockA an element that was already transposed by another thread. Note that this won't transpose anything.

#### b. If the code does not execute correctly for all BLOCK_SIZE values, what is the root cause of this incorrect execution behavior? Suggest a fix to the code to make it work for all BLOCK_SIZE values.

As mentioned above, the root cause is the lack of thread synchronization. To fix this, we can add a `__syncthreads()` call after the `blockA[threadIdx.y][threadIdx.x] = A_elements[baseIdx];` line. This will ensure that all threads have written their values to shared memory before any of them reads from it.


### 11.
#### a. How many versions of the variable i are there?
1024, one per thread. In registers.
#### b. How many versions of the array x[] are there?
Also 1024, for the same reason. In registers because of the constant access pattern (we assume the footnote in page 101 applies here).
#### c. How many versions of the variable y_s are there?
1 per block, so 8. In shared memory.
#### d. How many versions of the array b_s[] are there?
1 per block, so 8. In shared memory.
#### e. What is the amount of shared memory used per block (in bytes)?
y_s + b_s = 1 * 4 + 128 * 4 = 516 bytes. (assuming 32 bit floats)
#### f. What is the floating-point to global memory access ratio of the kernel (in OP/B)?
Each thread reads from global memory 5 elements: 4 from `a` (line 7) and 1 from `b` (line 12). It performs 10 floating point operations (lines 14, 15): 5 prods and 5 sums. So, flop to global memory access ratio is 10/5 = 2 ops per byte.

### 12.
#### a. The kernel uses 64 threads/block, 27 registers/thread, and 4 KB of shared memory/SM.
I'm not sure whether "4 KB of shared memory/SM" refers to total memory used by all blocks in the SM, or the memory used by a single block. I'll assume the latter. In this case, the limiting factor is shared memory: all 32 blocks of 64 threads could be scheduled on the SM, but would use 32 * 4 KB = 128 KB of shared memory, which is more than the 96 KB available.
#### b. The kernel uses 256 threads/block, 31 registers/thread, and 8 KB of shared memory/SM.
With the same assumption as before, there is no limiting factor and the kernel can achieve full occupancy.