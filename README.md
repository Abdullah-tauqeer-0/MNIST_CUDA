# MNIST-CUDA-C  
*A pure-C, GPU-accelerated LeNet-5 reference project*

---

## 1  Why this repo?

* **Pedagogical:** learn CUDA without leaning on C++ or Python.  
* **Minimal:** ~500 lines of host C + a single `.cu` file.  
* **Portable:** builds with GCC ≥ 9, CUDA 12.x, cuDNN 9.x on any modern NVIDIA GPU (CC ≥ 6.0).  
* **Extensible:** hooks for back-prop, mixed precision, TensorRT, multi-GPU.

---

## 2  Directory layout

```
.
├── Makefile             # one-line build: `make`
├── include/             # public C headers
│   ├── mnist.h          # tensor + network structs, loader prototypes
│   └── utils.h          # CUDA error macro + lightweight timers
├── src/
│   ├── dataloader.c     # IDX parsing & host-side normalisation
│   ├── kernels.cu       # tiny hand-written ReLU fwd/bwd
│   ├── layers.c         # cuDNN/cuBLAS helpers + forward_lenet()
│   └── main.c           # data load, weight init, inference timing
└── README.md
```

---

## 3  Prerequisites

| Item | Tested version | Notes |
|------|----------------|-------|
| **GPU** | GTX 1650 ➜ A100 | Compute Capability ≥ 6.0 |
| **Driver** | 550.xx | `nvidia-smi` should list your GPU |
| **CUDA Toolkit** | 12.5 | `/usr/local/cuda` in `PATH` |
| **cuDNN** | 9.1 | install `libcudnn9` + `libcudnn9-dev` |
| **GCC / Clang** | ≥ 9 / ≥ 12 | must accept `-std=c99` |

> **Windows?** Works with MSYS2 + `mingw64` GCC ≥ 13 and CUDA for Windows—change path separators in `Makefile`.

---

## 4  Dataset

Place the four raw IDX files **uncompressed** under `./data/` (default) or any folder you pass on the command line:

```
data/
 ├── train-images-idx3-ubyte
 ├── train-labels-idx1-ubyte
 ├── t10k-images-idx3-ubyte
 └── t10k-labels-idx1-ubyte
```

Download from <http://yann.lecun.com/exdb/mnist/> or any mirror.

---

## 5  Build & run

```bash
# clone
git clone https://github.com/yourname/mnist-cuda-c.git
cd mnist-cuda-c

# compile (≈ 2 s on laptop)
make

# run inference on the 10 k test set
./mnist                 # uses ./data by default
# or
./mnist /path/to/mnist  # custom dataset folder
```

Sample output on RTX 4060 Laptop:

```
Loaded 10000 test images (28x28)
Forward pass: 6.12 ms  (1634.0 images/s)
Accuracy (random weights): 10.57 %
```

---

## 6  Project walkthrough

| Step | Source | Key APIs | Learning outcome |
|------|--------|----------|------------------|
| 1. Parse IDX → float32 | `dataloader.c` | stdio | Big-endian handling, host normalisation |
| 2. GPU tensors | `mnist.h`, `layers.c` | `cudaMalloc`, `cudaMemcpy` | Manual memory management |
| 3. Conv/Pool | `layers.c` | `cudnnConvolutionForward`, `cudnnPoolingForward` | cuDNN descriptors & workspace |
| 4. FC layers | `layers.c` | `cublasSgemm`, `cublasSger` | Column-major GEMM, bias add trick |
| 5. Activation | `kernels.cu` | custom kernels | Grid/block sizing, C linkage |
| 6. Timing | `utils.h` | `cudaEventRecord` | Accurate kernel profiling |

---

## 7  Training (todo)

The forward path is complete; backward pass stubs are ready:

1. **Back-prop convolutions / pool:** use `cudnnConvolutionBackwardData`, `cudnnPoolingBackward`.  
2. **Back-prop fully-connected:** transpose arguments in `cublasSgemm`.  
3. **Optimizer:** write a tiny SGD/Adam kernel or call cuBLAS AXPY.  
4. **Loop:** mini-batch (64) for ~5 epochs → ≈ 99 % test accuracy in ~30 s on RTX 4060.

---

## 8  Extensions

* **Mixed precision:** switch tensors to `__half`, set cuBLAS math mode to `CUBLAS_TENSOR_OP_MATH`.  
* **TensorRT export:** write ONNX exporter or use `trtexec --onnx=<model>`.  
* **Multi-GPU:** data-parallel with NCCL `ncclAllReduce`, one process per device.  
* **FPGA/Metal:** port element-wise kernels first, then replace cuDNN/cuBLAS with vendor libs.

---

## 9  Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `CUDA error … invalid device function` | GPU too old (CC < 6.0) | Recompile with lower `-gencode`, or use newer GPU |
| cuDNN symbol not found | Wrong cuDNN version or missing `LD_LIBRARY_PATH` | `export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH` |
| Accuracy stays at 10 % after training | Labels not shuffled with images | Ensure batch shuffling pairs images & labels |


---

## 10  Acknowledgements

* Yann LeCun for MNIST  
* NVIDIA for cuDNN/cuBLAS sample code inspiration  
* Andrew Ng & CS231n notes for the original LeNet diagrams
