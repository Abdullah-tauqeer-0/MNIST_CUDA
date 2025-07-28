# ---- Compiler settings -------------------------------------------------
CC      = gcc
NVCC    = nvcc


# ---- Compiler settings -------------------------------------------------
CUDA_HOME ?= /usr/local/cuda

CFLAGS  = -O3 -std=c99 -Iinclude -I$(CUDA_HOME)/include
NVFLAGS = -O3 --std=c99 --host-compilation c --device-compilation c \
          -Iinclude -I$(CUDA_HOME)/include -lcudart -lcublas -lcudnn


# Host-side C (pure C99) and device-side CUDA flags
CFLAGS  = -O3 -std=c99 -Iinclude
NVFLAGS = -O3 --std=c99 --host-compilation c --device-compilation c \
          -Iinclude -lcudart -lcublas -lcudnn

# ---- Source & object lists --------------------------------------------
SRC_C   := $(wildcard src/*.c)
SRC_CU  := $(wildcard src/*.cu)
OBJ_C   := $(patsubst %.c,%.o,$(SRC_C))
OBJ_CU  := $(patsubst %.cu,%.o,$(SRC_CU))

# ---- Targets -----------------------------------------------------------
all: mnist                           # default target

mnist: $(OBJ_C) $(OBJ_CU)            # final executable
	$(NVCC) $(NVFLAGS) $^ -o $@

# Pattern rules: compile C → .o  and .cu → .o
src/%.o: src/%.c
	$(CC)  $(CFLAGS) -c $< -o $@

src/%.o: src/%.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@

clean:
	rm -f src/*.o mnist
.PHONY: all clean
