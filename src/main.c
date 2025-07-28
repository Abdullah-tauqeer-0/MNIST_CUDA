/* main.c : drive data loading, weight init, and one forward pass
 *          Compile & link via Makefile (pure-C, no C++).               */
 #include <stdio.h>
 #include <stdlib.h>
 #include <time.h>
 #include "mnist.h"
 #include "utils.h"
 
 /* ------------------------------------------------------------------ */
 /*  Quick uniform random [-r,+r]                                       */
 /* ------------------------------------------------------------------ */
 static void fill_rand(float *buf, size_t n, float r)
 {
     for (size_t i = 0; i < n; ++i)
         buf[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * r;
 }
 
 /* ------------------------------------------------------------------ */
 /*  Allocate + initialise all LeNet parameters on the GPU             */
 /* ------------------------------------------------------------------ */
 static void init_lenet(lenet_t *net)
 {
     /* define shapes first … */
     net->conv1_w = (tensor_t){ .n=6,  .c=1,  .h=5, .w=5 };
     net->conv1_b = (tensor_t){ .n=1,  .c=6,  .h=1, .w=1 };
     net->conv2_w = (tensor_t){ .n=16, .c=6,  .h=5, .w=5 };
     net->conv2_b = (tensor_t){ .n=1,  .c=16, .h=1, .w=1 };
     net->fc1_w   = (tensor_t){ .n=120, .c=256, .h=1, .w=1 }; /* 16*4*4=256 */
     net->fc1_b   = (tensor_t){ .n=1,   .c=120, .h=1, .w=1 };
     net->fc2_w   = (tensor_t){ .n=10,  .c=120, .h=1, .w=1 };
     net->fc2_b   = (tensor_t){ .n=1,   .c=10,  .h=1, .w=1 };
 
     tensor_t *all[] = {
         &net->conv1_w, &net->conv1_b, &net->conv2_w, &net->conv2_b,
         &net->fc1_w, &net->fc1_b, &net->fc2_w, &net->fc2_b, NULL
     };
 
     /* host buffer for random init then cudaMemcpy to each tensor */
     for (int i = 0; all[i]; ++i) {
         tensor_t *t = all[i];
         tensor_malloc(t);
 
         size_t elems = (size_t)t->n * t->c * t->h * t->w;
         float *tmp = malloc(elems * sizeof(float));
         fill_rand(tmp, elems, 0.05f);               /* Xavier-ish */
         CUDA_CHECK(cudaMemcpy(t->d, tmp,
                               elems * sizeof(float),
                               cudaMemcpyHostToDevice));
         free(tmp);
     }
 }
 
 /* ------------------------------------------------------------------ */
 int main(int argc, char **argv)
 {
     srand((unsigned)time(NULL));
 
     const char *root = (argc > 1) ? argv[1] : "./data";
     char img_path[256], lbl_path[256];
 
     snprintf(img_path, sizeof img_path, "%s/t10k-images-idx3-ubyte", root);
     snprintf(lbl_path, sizeof lbl_path, "%s/t10k-labels-idx1-ubyte",  root);
 
     /* -------- load test set into host RAM -------------------------- */
     float  *h_imgs = NULL;
     uint8_t *h_lbls = NULL;
     int n_imgs, rows, cols;
 
     if (!read_idx_images(img_path, &h_imgs, &n_imgs, &rows, &cols) ||
         !read_idx_labels(lbl_path, &h_lbls, &n_imgs)) {
         fprintf(stderr, "Failed to load MNIST from %s\n", root);
         return EXIT_FAILURE;
     }
     printf("Loaded %d test images (%dx%d)\n", n_imgs, rows, cols);
 
     /* -------- push images to GPU ---------------------------------- */
     tensor_t x = { .n=n_imgs, .c=1, .h=rows, .w=cols };
     tensor_malloc(&x);
     CUDA_CHECK(cudaMemcpy(x.d, h_imgs,
                           (size_t)n_imgs*rows*cols*sizeof(float),
                           cudaMemcpyHostToDevice));
 
     /* -------- network weights ------------------------------------- */
     lenet_t net;
     init_lenet(&net);
 
     /* -------- output tensor (N×10) -------------------------------- */
     tensor_t y = { .n=n_imgs, .c=10, .h=1, .w=1 };
     tensor_malloc(&y);
 
     layers_init();           /* set up cuDNN/cuBLAS descriptors */
 
     gpu_timer_t timer;
     timer_create(&timer); timer_start(&timer);
 
     forward_lenet(&x, &net, &y);
 
     float ms = timer_stop(&timer);
     printf("Forward pass: %.2f ms  (%.1f images/s)\n",
            ms, 1000.0f*n_imgs/ms);
     timer_destroy(&timer);
 
     /* -------- pull predictions back & compute top-1 --------------- */
     float *h_out = malloc((size_t)n_imgs*10*sizeof(float));
     CUDA_CHECK(cudaMemcpy(h_out, y.d,
                           (size_t)n_imgs*10*sizeof(float),
                           cudaMemcpyDeviceToHost));
 
     int correct = 0;
     for (int i = 0; i < n_imgs; ++i) {
         int best = 0;
         for (int j = 1; j < 10; ++j)
             if (h_out[i*10+j] > h_out[i*10+best]) best = j;
         if (best == h_lbls[i]) ++correct;
     }
     printf("Accuracy (random weights): %.2f %%\n",
            100.0f * correct / n_imgs);
 
     /* -------- tidy up --------------------------------------------- */
     free(h_imgs); free(h_lbls); free(h_out);
     tensor_free(&x); tensor_free(&y);
     /* weights */
     tensor_free(&net.conv1_w); tensor_free(&net.conv1_b);
     tensor_free(&net.conv2_w); tensor_free(&net.conv2_b);
     tensor_free(&net.fc1_w);   tensor_free(&net.fc1_b);
     tensor_free(&net.fc2_w);   tensor_free(&net.fc2_b);
 
     layers_shutdown();
     return 0;
 }