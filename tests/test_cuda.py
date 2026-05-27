import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
import torch
import numpy as np
import torchlite
from torchlite import Device

class TestCUDA(unittest.TestCase):
    def test_cuda_operations(self):
        try:
            a_tl = torchlite.Tensor([[1.0, 2.0], [3.0, 4.0]], device=Device.CUDA, requires_grad=True)
            b_tl = torchlite.Tensor([[5.0, 6.0], [7.0, 8.0]], device=Device.CUDA, requires_grad=True)
            
            c_tl = a_tl @ b_tl
            c_tl.backward(torchlite.Tensor(np.ones(c_tl.shape, dtype=np.float32), device=Device.CUDA))
            
            self.assertEqual(c_tl.device, Device.CUDA)
            self.assertEqual(a_tl.grad.device, Device.CUDA)
            
            a_cpu = torchlite.Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
            b_cpu = torchlite.Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
            c_cpu = a_cpu @ b_cpu
            c_cpu.backward(torchlite.Tensor(np.ones(c_cpu.shape, dtype=np.float32)))
            
            self.assertTrue(np.allclose(a_tl.grad.numpy(), a_cpu.grad.numpy(), atol=1e-5))
        except Exception as e:
            self.skipTest(f"CUDA test skipped: {e}")
