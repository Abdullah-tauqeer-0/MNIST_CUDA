#ifndef MNIST_H
#define MNIST_H

#include <stdint.h>
#include <cuda_runtime.h>

/* ----------------------------- Tensor --------------------------------- */
/* Struct for 4-D tensor wrapper in NCHW (batch, channels, height, width) */
typedef struct {
    float *d;          /* pointer on DEVICE (GPU) memory                  */
    int    n, c, h, w; /* shape                                           */
} tensor_t;

/* ----------------------------- Network params ------------------------- */
typedef struct {
    tensor_t conv1_w, conv1_b;   /* 6×1×5×5  +  6                         */
    tensor_t conv2_w, conv2_b;   /* 16×6×5×5 + 16                         */
    tensor_t fc1_w,   fc1_b;     /* 120×400   + 120                       */
    tensor_t fc2_w,   fc2_b;     /* 10×120    + 10                        */
} lenet_t;

/* ----------------------------- Data I/O --------------------------------*/
/* Read MNIST IDX files (big-endian format).                              *
 * Each function allocates host memory and returns element count via out *
 * parameters. Returns element-size (images) or 0 on failure.             */
int  read_idx_images(const char *path,
                     float **out, int *count,
                     int *rows, int *cols);

int  read_idx_labels(const char *path,
                     uint8_t **out, int *count);

/* -------------------------- Tensor helpers ---------------------------- */
void tensor_malloc(tensor_t *t);   /* cudaMalloc + zero-init              */
void tensor_free  (tensor_t *t);   /* cudaFree                            */

#endif /* MNIST_H */
