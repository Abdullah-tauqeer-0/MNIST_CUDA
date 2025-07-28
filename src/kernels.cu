/* kernels.cu
 * ───────────
 * All hand-written GPU kernels live here.  We keep only simple element-wise
 * operations in CUDA and leave heavy work (conv, GEMM) to cuDNN / cuBLAS.
 *
 * IMPORTANT: we compile this file with
 *   nvcc --std=c99 --host-compilation c --device-compilation c
 * so we must stick to C-style syntax (no <iostream>, no new/delete, etc.).
 */
 #include <cuda_runtime.h>

 /* --------------------------------------------------------------------- */
 /*  ReLU forward: y = max(0, x)                                           */
 /* --------------------------------------------------------------------- */
 __global__ void relu_forward_kernel(float *x, int n)
 {
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
     if (idx < n) x[idx] = x[idx] > 0.0f ? x[idx] : 0.0f;
 }
 
 /* --------------------------------------------------------------------- */
 /*  ReLU backward: dx = (x > 0) ? dy : 0                                  */
 /* --------------------------------------------------------------------- */
 __global__ void relu_backward_kernel(const float *dy,
                                      const float *x,
                                      float *dx,
                                      int n)
 {
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
     if (idx < n) dx[idx] = (x[idx] > 0.0f) ? dy[idx] : 0.0f;
 }
 
 /* ------------------------------ Wrappers ----------------------------- */
 /*  These expose plain-C symbols the host code can call.                 */
 extern "C" {
 
 void relu_forward_cuda(float *d_x, int n)
 {
     const int BS = 256;
     int GS = (n + BS - 1) / BS;
     relu_forward_kernel<<<GS, BS>>>(d_x, n);
 }
 
 void relu_backward_cuda(const float *d_dy,
                         const float *d_x,
                         float *d_dx,
                         int n)
 {
     const int BS = 256;
     int GS = (n + BS - 1) / BS;
     relu_backward_kernel<<<GS, BS>>>(d_dy, d_x, d_dx, n);
 }
 
 } /* extern "C" */
 