# TorchLite

**TorchLite** is a minimal, lightweight, GPU-accelerated tensor library built from scratch. It bridges a high-level Python API with a low-level C++ and CUDA backend, providing fundamental tensor operations and a built-in Autograd engine.

## Features

- **C++ & CUDA Backend**: Fast execution of mathematical operations using custom CUDA kernels.
- **Python Frontend (PyBind11)**: A user-friendly Python interface that mimics the look and feel of modern Deep Learning frameworks (like PyTorch).
- **CPU & GPU Memory Management**: Seamlessly transfer tensors between CPU and GPU memory using `.to(Device.CUDA)` or `.to(Device.CPU)`.
- **Dynamic Autograd**: Tracks operations dynamically and builds a computational graph for backpropagation (`.backward()`).
- **Basic Operations**: Element-wise Addition, Subtraction, Multiplication, and Matrix Multiplication (`@`).

## Requirements

- CMake (>= 3.18)
- C++17 compatible compiler (GCC, Clang)
- NVIDIA CUDA Toolkit (nvcc)
- Python 3.x
- `numpy` (for testing and data ingestion)

## Installation

You can build the PyBind11 C++ extension using CMake.

```bash
# Clone the repository
git clone https://github.com/siddarth777/TorchLite.git
cd TorchLite

# Build the C++/CUDA backendc
cmake -B build -S .
cmake --build build -j
```

Once built, you can use the library from any Python script in the parent directory or by installing it into your virtual environment.

## Usage

```python
import torchlite
from torchlite import Device

# Create tensors (CPU by default)
a = torchlite.Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
b = torchlite.Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

# Matrix Multiplication
c = a @ b

# Element-wise Addition and Multiplication
d = c + a
e = d * b

# Backpropagation
# We pass a gradient of 1s to start the chain rule
e.backward(torchlite.Tensor([[1.0, 1.0], [1.0, 1.0]]))

# View the gradients calculated by the Autograd engine
print("Gradient of A:")
print(a.grad.numpy())

print("Gradient of B:")
print(b.grad.numpy())
```

### Using the GPU

To accelerate your operations on the GPU, simply pass `device=Device.CUDA` during instantiation or use `.to()`:

```python
# Create directly on GPU
a_gpu = torchlite.Tensor([[1.0, 2.0]], device=Device.CUDA, requires_grad=True)

# Or move an existing tensor
b_cpu = torchlite.Tensor([[3.0], [4.0]], requires_grad=True)
b_gpu = b_cpu.to(Device.CUDA)

# Operations run entirely on the CUDA device
c_gpu = a_gpu @ b_gpu
```

## Running Tests

To verify that TorchLite's forward passes and gradient calculations perfectly match PyTorch:

```bash
# Set up a virtual environment and install dependencies
python3 -m venv venv
./venv/bin/pip install torch numpy

# Run the comparative tests
./venv/bin/python tests/run_tests.py
```
