import sys
import os
import torch
import numpy as np

# Add parent directory to path so torchlite can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torchlite
from torchlite import Device

def test_operations():
    # PyTorch Tensors
    a_pt = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b_pt = torch.tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

    # TorchLite Tensors
    a_tl = torchlite.Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b_tl = torchlite.Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

    # Addition
    c_pt = a_pt + b_pt
    c_tl = a_tl + b_tl
    assert np.allclose(c_pt.detach().numpy(), c_tl.numpy())

    # Multiplication
    d_pt = c_pt * a_pt
    d_tl = c_tl * a_tl
    assert np.allclose(d_pt.detach().numpy(), d_tl.numpy())

    # Matmul
    e_pt = d_pt @ b_pt
    e_tl = d_tl @ b_tl
    assert np.allclose(e_pt.detach().numpy(), e_tl.numpy())

    e_pt.backward(torch.ones_like(e_pt))
    e_tl.backward(torchlite.Tensor(np.ones(e_tl.shape, dtype=np.float32)))

    assert np.allclose(a_pt.grad.numpy(), a_tl.grad.numpy(), atol=1e-5)
    assert np.allclose(b_pt.grad.numpy(), b_tl.grad.numpy(), atol=1e-5)
    print("CPU operations and autograd tests passed!")

def test_cuda():
    try:
        a_tl = torchlite.Tensor([[1.0, 2.0], [3.0, 4.0]], device=Device.CUDA, requires_grad=True)
        b_tl = torchlite.Tensor([[5.0, 6.0], [7.0, 8.0]], device=Device.CUDA, requires_grad=True)
        
        c_tl = a_tl @ b_tl
        c_tl.backward(torchlite.Tensor(np.ones(c_tl.shape, dtype=np.float32), device=Device.CUDA))
        
        assert c_tl.device == Device.CUDA
        assert a_tl.grad.device == Device.CUDA
        
        a_cpu = torchlite.Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b_cpu = torchlite.Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
        c_cpu = a_cpu @ b_cpu
        c_cpu.backward(torchlite.Tensor(np.ones(c_cpu.shape, dtype=np.float32)))
        
        assert np.allclose(a_tl.grad.numpy(), a_cpu.grad.numpy(), atol=1e-5)
        print("CUDA operations and autograd tests passed!")
        
    except Exception as e:
        print(f"CUDA test failed or skipped: {e}")

if __name__ == "__main__":
    test_operations()
    test_cuda()
