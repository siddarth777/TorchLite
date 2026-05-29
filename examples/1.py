import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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

a_gpu = torchlite.Tensor([[1.0, 2.0]], device=Device.CUDA, requires_grad=True)

# Or move an existing tensor
b_cpu = torchlite.Tensor([[3.0], [4.0]], requires_grad=True)
b_gpu = b_cpu.to(Device.CUDA)

# Operations run entirely on the CUDA device
c_gpu = a_gpu @ b_gpu

print(c_gpu.numpy());
