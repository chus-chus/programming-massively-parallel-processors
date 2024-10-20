## 4. Compute architecture and scheduling

#### 1.
**a.**
128/32 = 4

**b.**
8*4 = 32

**c.**  
i. 31  
ii. 2  
iii. 100%  
iv. (32-8)/32 = 75%  
v. Same as iv.

**d.**  
i. 32 (all)  
ii. Same as i.  
iii. 16/32 = 50%  

**e.**
There is a repeating pattern of different loop lengths: 5, 4, 3. A third of all threads in a warp will have 5 iterations in the loop, another third 4 and the last one 3. Thus, all warps have thread divergence.

#### 2.
2048.

#### 3.
2.

#### 4.
4,1 μs wait / 19.9 μs total = 20.6% of the time is spent waiting.

#### 5.
One shouldn’t assume that all threads in a warp execute with the same execution timing, so this could still cause them to have sync problems.

#### 6.
c.

#### 7.
All assignments are possible.  
**a.** 50%  
**b.** 50%  
**c.** 50%  
**d.** 100%  
**e.** 100%  

#### 8.
**a.** Yes.  
**b.** Not enough block slots.  
**c.** Not enough registers.  


#### 9.
Surprise because each supposed block has 1024 threads and the device limit is 512.