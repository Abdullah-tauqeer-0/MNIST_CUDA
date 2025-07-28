/* dataloader.c : read IDX-format MNIST files into host memory --------- */
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include "mnist.h"

/* convert big-endian 32-bit value to host byte order */
static uint32_t be32(const unsigned char b[4])
{
    return (uint32_t)b[0] << 24 |
           (uint32_t)b[1] << 16 |
           (uint32_t)b[2] <<  8 |
           (uint32_t)b[3];
}

/* --------------------------------------------------------------------- */
/*  Images : returns element-size (rows*cols) or 0 on failure            */
/* --------------------------------------------------------------------- */
int read_idx_images(const char *path,
                    float **out, int *count,
                    int *rows, int *cols)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) { perror(path); return 0; }

    unsigned char hdr[16];
    if (fread(hdr, 1, 16, fp) != 16) { fclose(fp); return 0; }

    /* header layout per MNIST spec */
    if (be32(hdr) != 0x00000803) {          /* magic number 2051 decimal */
        fprintf(stderr, "Not an IDX image file: %s\n", path);
        fclose(fp); return 0;
    }
    *count = (int)be32(hdr + 4);
    *rows  = (int)be32(hdr + 8);
    *cols  = (int)be32(hdr + 12);
    const int elem = (*rows) * (*cols);

    size_t nbytes = (size_t)(*count) * elem;
    unsigned char *tmp = malloc(nbytes);
    if (!tmp) { fclose(fp); return 0; }

    if (fread(tmp, 1, nbytes, fp) != nbytes) {
        fprintf(stderr, "Truncated image file\n");
        free(tmp); fclose(fp); return 0;
    }
    fclose(fp);

    /* allocate float buffer & normalise to [0,1] */
    *out = malloc(nbytes * sizeof(float));
    for (size_t i = 0; i < nbytes; ++i) (*out)[i] = tmp[i] / 255.0f;
    free(tmp);
    return elem;
}

/* --------------------------------------------------------------------- */
/*  Labels : 10-class uint8_t array                                      */
/* --------------------------------------------------------------------- */
int read_idx_labels(const char *path,
                    uint8_t **out, int *count)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) { perror(path); return 0; }

    unsigned char hdr[8];
    if (fread(hdr, 1, 8, fp) != 8) { fclose(fp); return 0; }

    if (be32(hdr) != 0x00000801) {          /* magic number 2049 */
        fprintf(stderr, "Not an IDX label file: %s\n", path);
        fclose(fp); return 0;
    }
    *count = (int)be32(hdr + 4);

    *out = malloc(*count);
    if (!*out) { fclose(fp); return 0; }

    if (fread(*out, 1, *count, fp) != (size_t)*count) {
        fprintf(stderr, "Truncated label file\n");
        free(*out); fclose(fp); return 0;
    }
    fclose(fp);
    return 1;   /* element size for labels (always 1) */
}
