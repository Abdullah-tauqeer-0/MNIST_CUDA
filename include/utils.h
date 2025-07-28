#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <cuda_runtime.h>

/* --------------------------------------------------------------------- */
/*  Error-checking helper                                                */
/* --------------------------------------------------------------------- */
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t _e = (call);                                             \
        if (_e != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                        \
                    __FILE__, __LINE__, cudaGetErrorString(_e));             \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

/* --------------------------------------------------------------------- */
/*  Light-weight GPU timer                                               */
/* --------------------------------------------------------------------- */
typedef struct {
    cudaEvent_t beg, end;
} gpu_timer_t;

/* create two CUDA events (lazy—no flags) */
static inline void timer_create(gpu_timer_t *t)
{
    CUDA_CHECK(cudaEventCreate(&t->beg));
    CUDA_CHECK(cudaEventCreate(&t->end));
}

static inline void timer_destroy(gpu_timer_t *t)
{
    CUDA_CHECK(cudaEventDestroy(t->beg));
    CUDA_CHECK(cudaEventDestroy(t->end));
}

static inline void timer_start(gpu_timer_t *t)
{
    CUDA_CHECK(cudaEventRecord(t->beg, 0));
}

static inline float timer_stop(gpu_timer_t *t)
/* returns elapsed milliseconds */
{
    CUDA_CHECK(cudaEventRecord(t->end, 0));
    CUDA_CHECK(cudaEventSynchronize(t->end));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t->beg, t->end));
    return ms;
}

#endif /* UTILS_H */
