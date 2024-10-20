## 3. Multidimensional grids and data

#### 1.
(todo insert code)  
**a.** w.r.t. to the configuration, we have a grid dimension of (1, ceil(n / nthreads), 1) because we have threads in the vertical dimension only: one will output each row.

w.r.t. the kernel, each thread has an internal for corresponding to the column dimension.

**b.** w.r.t. to the configuration, we have a grid dimension of (ceil(n / nthreads), 1, 1) because we have threads in the horizontal dimension only: one will output all columns.

w.r.t. the kernel, each thread has an internal for corresponding to the row dimension.

**c.** These solutions are good for when the matrix is really big and will not fit into memory. For a square matrix, both solutions are equal. When the output matrix has n > m (n = rows, m = columns), solution A is best. When m > n, solution B is best.

#### 2.
(todo insert code) 

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

