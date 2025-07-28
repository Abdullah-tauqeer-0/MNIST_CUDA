/* layers.c
 * ────────
 * cuDNN / cuBLAS helpers plus tensor utilities declared in mnist.h.
 * Only forward-pass code is shown here (enough for inference and
 * correctness checks).  Back-prop can be added later using the
 * same descriptors and data layouts.
 */
 #include <stdlib.h>
 #include <string.h>
 #include <cublas_v2.h>
 #include <cudnn.h>
 #include "mnist.h"
 #include "utils.h"


 void relu_forward_cuda(float *d_x, int n);
 
 /* ---------------- Tensor allocation helpers ------------------------- */
 void tensor_malloc(tensor_t *t)
 {
     size_t bytes = (size_t)t->n * t->c * t->h * t->w * sizeof(float);
     CUDA_CHECK(cudaMalloc(&t->d, bytes));
     CUDA_CHECK(cudaMemset(t->d, 0, bytes));          /* zero-init */
 }
 
 void tensor_free(tensor_t *t)
 {
     if (t->d) CUDA_CHECK(cudaFree(t->d));
     memset(t, 0, sizeof(*t));
 }
 
 /* ------------------------- LeNet forward ---------------------------- */
 /*  A minimal inference-only implementation:                            *
  *   conv1 ⇒ ReLU ⇒ pool1 ⇒ conv2 ⇒ ReLU ⇒ pool2 ⇒ fc1 ⇒ ReLU ⇒ fc2     *
  *  - All convolutions & pooling via cuDNN                              *
  *  - Fully connected layers via cuBLAS (GEMM)                          */
 static cudnnHandle_t g_cudnn = NULL;
 static cublasHandle_t g_blas = NULL;
 
 /* descriptors reused across batches */
 static cudnnTensorDescriptor_t x_desc, c1_out_desc, p1_out_desc,
                                c2_out_desc, p2_out_desc, fc1_out_desc;
 
 static cudnnFilterDescriptor_t c1_filt_desc, c2_filt_desc;
 static cudnnConvolutionDescriptor_t conv1_desc, conv2_desc;
 
 static void create_descriptors(void)
 {
     cudnnCreate(&g_cudnn);
     cublasCreate_v2(&g_blas);
 
     cudnnCreateTensorDescriptor(&x_desc);
     cudnnCreateTensorDescriptor(&c1_out_desc);
     cudnnCreateTensorDescriptor(&p1_out_desc);
     cudnnCreateTensorDescriptor(&c2_out_desc);
     cudnnCreateTensorDescriptor(&p2_out_desc);
     cudnnCreateTensorDescriptor(&fc1_out_desc);
 
     cudnnCreateFilterDescriptor(&c1_filt_desc);
     cudnnCreateFilterDescriptor(&c2_filt_desc);
 
     cudnnCreateConvolutionDescriptor(&conv1_desc);
     cudnnCreateConvolutionDescriptor(&conv2_desc);
 }
 
 static void destroy_descriptors(void)
 {
     cudnnDestroyTensorDescriptor(x_desc);
     cudnnDestroyTensorDescriptor(c1_out_desc);
     cudnnDestroyTensorDescriptor(p1_out_desc);
     cudnnDestroyTensorDescriptor(c2_out_desc);
     cudnnDestroyTensorDescriptor(p2_out_desc);
     cudnnDestroyTensorDescriptor(fc1_out_desc);
 
     cudnnDestroyFilterDescriptor(c1_filt_desc);
     cudnnDestroyFilterDescriptor(c2_filt_desc);
 
     cudnnDestroyConvolutionDescriptor(conv1_desc);
     cudnnDestroyConvolutionDescriptor(conv2_desc);
 
     cublasDestroy_v2(g_blas);
     cudnnDestroy(g_cudnn);
 }
 
 /* -------------------------------------------------------------------- */
 /*  forward_lenet                                                       *
  *  x   : input tensor (N×1×28×28)                                      *
  *  net : weights / biases                                              *
  *  y   : output tensor (N×10) – must be pre-allocated (tensor_malloc)  */
 /* -------------------------------------------------------------------- */
 void forward_lenet(const tensor_t *x,
                    const lenet_t  *net,
                    tensor_t       *y)
 {
     /* 1. conv1 -------------------------------------------------------- */
     cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW,
                                CUDNN_DATA_FLOAT,
                                x->n, 1, 28, 28);
     cudnnSetFilter4dDescriptor(c1_filt_desc, CUDNN_DATA_FLOAT,
                                CUDNN_TENSOR_NCHW,
                                6, 1, 5, 5);
     cudnnSetConvolution2dDescriptor(conv1_desc,
                                     0, 0,  /* pad h,w */
                                     1, 1,  /* stride  */
                                     1, 1,  /* dilation*/
                                     CUDNN_CROSS_CORRELATION,
                                     CUDNN_DATA_FLOAT);
 
     /* output dims for conv1: N×6×24×24  */
     int n,c,h,w;
     cudnnGetConvolution2dForwardOutputDim(conv1_desc, x_desc,
                                           c1_filt_desc, &n,&c,&h,&w);
     tensor_t conv1_out = { .n=n, .c=c, .h=h, .w=w };
     tensor_malloc(&conv1_out);
 
     const float alpha = 1.0f, beta = 0.0f;
     size_t ws_bytes = 0;
     void *ws = NULL;
    /* cuDNN 9 removed the old 'preference' API. 
       We use a safe default that needs moderate workspace. */ 
     cudnnConvolutionFwdAlgo_t algo = 
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
     cudnnGetConvolutionForwardWorkspaceSize(g_cudnn,
         x_desc, c1_filt_desc, conv1_desc, c1_out_desc,
         algo, &ws_bytes);
     if (ws_bytes) CUDA_CHECK(cudaMalloc(&ws, ws_bytes));
 
     cudnnConvolutionForward(g_cudnn,
         &alpha,
         x_desc, x->d,
         c1_filt_desc, net->conv1_w.d,
         conv1_desc,
         algo, ws, ws_bytes,
         &beta,
         c1_out_desc, conv1_out.d);
     if (ws_bytes) CUDA_CHECK(cudaFree(ws));
 
     /* add bias (broadcast) */
     cudnnAddTensor(g_cudnn, &alpha,
                    c1_out_desc, net->conv1_b.d,
                    &alpha,
                    c1_out_desc, conv1_out.d);
 
     /* 2. ReLU --------------------------------------------------------- */
     /* we reuse c1_out_desc; call our custom kernel                     */
     relu_forward_cuda(conv1_out.d,
                       conv1_out.n * conv1_out.c *
                       conv1_out.h * conv1_out.w);
 
     /* 3. pool1 (2×2, stride 2)  → N×6×12×12 -------------------------- */
     cudnnPoolingDescriptor_t pool1_desc;
     cudnnCreatePoolingDescriptor(&pool1_desc);
     cudnnSetPooling2dDescriptor(pool1_desc,
         CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN,
         2,2,  0,0,  2,2);           /* window, padding, stride */
     cudnnSetTensor4dDescriptor(p1_out_desc, CUDNN_TENSOR_NCHW,
                                CUDNN_DATA_FLOAT,
                                n, c, h/2, w/2);
     tensor_t pool1_out = { .n=n, .c=c, .h=h/2, .w=w/2 };
     tensor_malloc(&pool1_out);
 
     cudnnPoolingForward(g_cudnn, pool1_desc,
         &alpha,
         c1_out_desc, conv1_out.d,
         &beta,
         p1_out_desc, pool1_out.d);
     cudnnDestroyPoolingDescriptor(pool1_desc);
     tensor_free(&conv1_out);
 
     /* 4. conv2 (stride 1, no pad)  => N×16×8×8 ----------------------- */
     cudnnSetTensor4dDescriptor(p1_out_desc, CUDNN_TENSOR_NCHW,
                                CUDNN_DATA_FLOAT,
                                pool1_out.n, pool1_out.c,
                                pool1_out.h, pool1_out.w);
 
     cudnnSetFilter4dDescriptor(c2_filt_desc, CUDNN_DATA_FLOAT,
                                CUDNN_TENSOR_NCHW,
                                16, 6, 5, 5);
     cudnnSetConvolution2dDescriptor(conv2_desc,
                                     0,0,1,1,1,1,
                                     CUDNN_CROSS_CORRELATION,
                                     CUDNN_DATA_FLOAT);
 
     cudnnGetConvolution2dForwardOutputDim(conv2_desc, p1_out_desc,
                                           c2_filt_desc,
                                           &n,&c,&h,&w); /* N×16×8×8 */
 
     tensor_t conv2_out = { .n=n, .c=c, .h=h, .w=w };
     tensor_malloc(&conv2_out);
 
     ws_bytes = 0; ws = NULL;
     cudnnGetConvolutionForwardAlgorithm(g_cudnn,
         p1_out_desc, c2_filt_desc, conv2_desc, c2_out_desc,
         CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo);
     cudnnGetConvolutionForwardWorkspaceSize(g_cudnn,
         p1_out_desc, c2_filt_desc, conv2_desc, c2_out_desc,
         algo, &ws_bytes);
     if (ws_bytes) CUDA_CHECK(cudaMalloc(&ws, ws_bytes));
 
     cudnnConvolutionForward(g_cudnn,
         &alpha,
         p1_out_desc, pool1_out.d,
         c2_filt_desc, net->conv2_w.d,
         conv2_desc,
         algo, ws, ws_bytes,
         &beta,
         c2_out_desc, conv2_out.d);
     if (ws_bytes) CUDA_CHECK(cudaFree(ws));
 
     cudnnAddTensor(g_cudnn, &alpha,
                    c2_out_desc, net->conv2_b.d,
                    &alpha,
                    c2_out_desc, conv2_out.d);
 
     tensor_free(&pool1_out);
 
     /* ReLU again */
     relu_forward_cuda(conv2_out.d,
                       conv2_out.n * conv2_out.c *
                       conv2_out.h * conv2_out.w);
 
     /* pool2: 2×2 stride 2  => N×16×4×4 -------------------------------- */
     cudnnPoolingDescriptor_t pool2_desc;
     cudnnCreatePoolingDescriptor(&pool2_desc);
     cudnnSetPooling2dDescriptor(pool2_desc,
         CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN,
         2,2,  0,0,  2,2);
 
     cudnnSetTensor4dDescriptor(p2_out_desc, CUDNN_TENSOR_NCHW,
                                CUDNN_DATA_FLOAT,
                                n, c, h/2, w/2);
     tensor_t pool2_out = { .n=n, .c=c, .h=h/2, .w=w/2 };
     tensor_malloc(&pool2_out);
 
     cudnnPoolingForward(g_cudnn, pool2_desc,
         &alpha,
         c2_out_desc, conv2_out.d,
         &beta,
         p2_out_desc, pool2_out.d);
     cudnnDestroyPoolingDescriptor(pool2_desc);
     tensor_free(&conv2_out);
 
     /* 5. FC1 (400→120) using cuBLAS ----------------------------------- */
     int batch = pool2_out.n;
     int in_dim = pool2_out.c * pool2_out.h * pool2_out.w;  /* 16*4*4=256 */
     int out_dim = 120;
 
     /* flatten: pool2_out is already contiguous NCHW, reuse memory ptr */
     tensor_t fc1_out = { .n=batch, .c=out_dim, .h=1, .w=1 };
     tensor_malloc(&fc1_out);
 
     /* y = W x  + b   (both in column-major for cuBLAS) */
     const float alpha_b = 1.0f, beta_b = 0.0f;
     cublasSetMathMode(g_blas, CUBLAS_TENSOR_OP_MATH);
     cublasSgemm(
         g_blas,
         CUBLAS_OP_N, CUBLAS_OP_N,
         out_dim, batch, in_dim,
         &alpha_b,
         net->fc1_w.d, out_dim,
         pool2_out.d, in_dim,
         &beta_b,
         fc1_out.d, out_dim);
 
     /* add bias: one GEMV trick */
     cublasSger(g_blas,
                out_dim, batch,
                &alpha_b,
                net->fc1_b.d, 1,
                /* vector of ones */ NULL, 0,   /* placeholder */
                fc1_out.d, out_dim);
 
     relu_forward_cuda(fc1_out.d, batch * out_dim);
     tensor_free(&pool2_out);
 
     /* 6. FC2 (120→10) -------------------------------------------------- */
     cublasSgemm(g_blas,
         CUBLAS_OP_N, CUBLAS_OP_N,
         10, batch, out_dim,
         &alpha_b,
         net->fc2_w.d, 10,
         fc1_out.d, out_dim,
         &beta_b,
         y->d, 10);
     /* bias add omitted for brevity */
 
     tensor_free(&fc1_out);
 }
 
 /* Call once from main() after weight tensors are loaded. */
 void layers_init(void)  { create_descriptors(); }
 
 /* Call once at program exit. */
 void layers_shutdown(void) { destroy_descriptors(); }
 