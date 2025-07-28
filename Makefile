CC      = gcc
NVCC    = nvcc
CFLAGS  = -O3 -std=c99 -Iinclude
NVFLAGS = -O3 --std=c99 --host-compilation c --device-compilation c \
          -Iinclude -lcudart -lcublas -lcudnn

SRC_C   = $(wildcard src/*.c)
SRC_CU  = src/kernels.cu
OBJ_C   = $(SRC_C:.c=.o)
OBJ_CU  = $(SRC_CU:.cu=.o)

all: mnist

mnist: $(OBJ_C) $(OBJ_CU)
	$(NVCC) $(NVFLAGS) $^ -o $@

clean:
	rm -f src/*.o mnist
