# matrix-mul-pytorch-cuda-cpu-analysis
# Matrix Multiplication Performance Comparison on CPU and GPU


Comparing the performance of matrix multiplication on a CPU and a GPU using PyTorch. The primary goal is to demonstrate the significant performance improvement achievable when leveraging PyTorch’s built-in GPU acceleration capabilities for large-scale linear algebra operations, such as matrix multiplication.

## **Overview**

Matrix multiplication is a core operation in various computational tasks, especially in fields like data science, machine learning, and scientific computing. PyTorch provides a straightforward way to utilize GPU acceleration without directly writing CUDA code, making it easier to enhance performance for matrix operations.

## **Test Environment**

- **GPU Device**: Tesla T4
- **CPU**: The specific CPU model is not detailed, but the performance timings are provided.
- **Libraries Used**: 
  - PyTorch
  - Matplotlib (for plotting, if used)

## **Performance Results**

### **CPU Performance**

| Matrix Size | Time Taken (CPU) |
|-------------|------------------|
| 100x100     | 0.0002 seconds   |
| 500x500     | 0.0025 seconds   |
| 1000x1000   | 0.0187 seconds   |
| 2000x2000   | 0.1527 seconds   |
| 3000x3000   | 0.7347 seconds   |
| 4000x4000   | 2.6599 seconds   |
| 5000x5000   | 3.8251 seconds   |
| 10000x10000 | 16.3859 seconds  |
| 20000x20000 | 129.6678 seconds |

![CPU Performance](/cpu.png)


### **GPU Performance**

| Matrix Size | Time Taken (GPU) |
|-------------|------------------|
| 100x100     | 0.0001 seconds   |
| 500x500     | 0.0001 seconds   |
| 1000x1000   | 0.0001 seconds   |
| 2000x2000   | 0.0001 seconds   |
| 3000x3000   | 0.0001 seconds   |
| 4000x4000   | 0.0000 seconds   |
| 5000x5000   | 0.0000 seconds   |
| 10000x10000 | 0.0001 seconds   |
| 20000x20000 | 0.0023 seconds   |

![GPU Performance](/gpu.png)


## **Key Observations**

- **Significant Speedup**: The GPU (Tesla T4) demonstrates a remarkable speedup compared to the CPU, especially as matrix sizes increase.
- **Efficient GPU Utilization**: PyTorch automatically uses GPU acceleration for operations on tensors that are moved to the GPU, simplifying the process of leveraging hardware acceleration.
- **Exponential Increase in CPU Time**: CPU computation times increase significantly with matrix size, illustrating the limitations of CPU-only computations for large-scale tasks.

### **Prerequisites**

- Python 3.x
- PyTorch with CUDA support
- Matplotlib (optional, for plotting results)

