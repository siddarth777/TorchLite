import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
import torch
import numpy as np
import torchlite

class TestSub(unittest.TestCase):
    def test_sub_forward_backward(self):
        a_pt = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b_pt = torch.tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
        
        a_tl = torchlite.Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b_tl = torchlite.Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
        
        c_pt = a_pt - b_pt
        c_tl = a_tl - b_tl
        
        self.assertTrue(np.allclose(c_pt.detach().numpy(), c_tl.numpy()))
        
        c_pt.backward(torch.ones_like(c_pt))
        c_tl.backward(torchlite.Tensor(np.ones(c_tl.shape, dtype=np.float32)))
        
        self.assertTrue(np.allclose(a_pt.grad.numpy(), a_tl.grad.numpy(), atol=1e-5))
        self.assertTrue(np.allclose(b_pt.grad.numpy(), b_tl.grad.numpy(), atol=1e-5))
