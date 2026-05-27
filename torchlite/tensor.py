import numpy as np
import os
import sys

# Add the build directory to the path so we can import the C++ extension
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'build')))

try:
    import _C
    from _C import Device
except ImportError as e:
    print(f"Failed to import C++ backend: {e}")
    class Device:
        CPU = 0
        CUDA = 1

class Context:
    """Stores information for the backward pass."""
    def __init__(self):
        self.saved_tensors = ()
        
    def save_for_backward(self, *args):
        self.saved_tensors = args

class Function:
    """Base class for Autograd functions."""
    @classmethod
    def apply(cls, *args, **kwargs):
        ctx = Context()
        result = cls.forward(ctx, *args, **kwargs)
        
        requires_grad = any(isinstance(arg, Tensor) and arg.requires_grad for arg in args)
        if requires_grad:
            result.requires_grad = True
            result.grad_fn = cls
            result.ctx = ctx
            result.parents = [arg for arg in args if isinstance(arg, Tensor)]
        
        return result

class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        out_c = a._tensor.add(b._tensor)
        return Tensor(out_c, device=a.device, requires_grad=False)
        
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        return grad_output, grad_output

class Sub(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        out_c = a._tensor.sub(b._tensor)
        return Tensor(out_c, device=a.device, requires_grad=False)
        
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        # Grad w.r.t b is -grad_output. Need a tensor of -1s.
        neg_ones = Tensor(np.ones(grad_output.shape, dtype=np.float32) * -1.0, device=grad_output.device)
        neg_grad_output = Tensor(grad_output._tensor.mul(neg_ones._tensor), device=grad_output.device)
        return grad_output, neg_grad_output

class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        out_c = a._tensor.mul(b._tensor)
        return Tensor(out_c, device=a.device, requires_grad=False)
        
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = Tensor(grad_output._tensor.mul(b._tensor), device=grad_output.device)
        grad_b = Tensor(grad_output._tensor.mul(a._tensor), device=grad_output.device)
        return grad_a, grad_b

class Matmul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        out_c = a._tensor.matmul(b._tensor)
        return Tensor(out_c, device=a.device, requires_grad=False)
        
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = Tensor(grad_output._tensor.matmul(b._tensor.transpose()), device=grad_output.device)
        grad_b = Tensor(a._tensor.transpose().matmul(grad_output._tensor), device=grad_output.device)
        return grad_a, grad_b

class Tensor:
    def __init__(self, data, shape=None, device=None, requires_grad=False):
        if device is None:
            device = Device.CPU
            
        self.device = device
        self.requires_grad = requires_grad
        self.grad = None
        self.grad_fn = None
        self.ctx = None
        self.parents = []
        
        if isinstance(data, _C.Tensor):
            self._tensor = data
        elif isinstance(data, (list, tuple)):
            arr = np.array(data, dtype=np.float32)
            self._tensor = _C.Tensor(arr.flatten().tolist(), list(arr.shape), device)
        elif isinstance(data, np.ndarray):
            arr = data.astype(np.float32)
            self._tensor = _C.Tensor(arr.flatten().tolist(), list(arr.shape), device)
        elif isinstance(data, (int, float)):
            arr = np.array([data], dtype=np.float32)
            self._tensor = _C.Tensor(arr.flatten().tolist(), list(arr.shape), device)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
            
    @property
    def shape(self):
        return tuple(self._tensor.shape)
        
    def to(self, device):
        if self.device == device:
            return self
        out_c = self._tensor.to(device)
        # Does not carry over grad history to new device automatically in this simple autograd
        return Tensor(out_c, device=device, requires_grad=self.requires_grad)
        
    def to_list(self):
        flat = self._tensor.to_list()
        # Ensure we return a nested list if shape > 1D
        arr = np.array(flat).reshape(self.shape)
        if arr.size == 1:
            return flat[0]
        return arr.tolist()
        
    def numpy(self):
        return np.array(self._tensor.to_list(), dtype=np.float32).reshape(self.shape)

    def backward(self, grad=None):
        if not self.requires_grad:
            raise RuntimeError("Tensor does not require grad")
            
        if grad is None:
            if self.shape == () or self.shape == (1,):
                grad = Tensor([1.0], shape=self.shape, device=self.device)
            else:
                raise RuntimeError("Grad must be specified for non-scalar tensors")
                
        self.grad = grad
        
        # Build topological order
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for parent in v.parents:
                    build_topo(parent)
                topo.append(v)
                
        build_topo(self)
        
        # Backprop
        for v in reversed(topo):
            if v.grad_fn is not None:
                grads = v.grad_fn.backward(v.ctx, v.grad)
                if not isinstance(grads, tuple):
                    grads = (grads,)
                    
                for parent, g in zip(v.parents, grads):
                    if parent.requires_grad:
                        if parent.grad is None:
                            parent.grad = g
                        else:
                            parent.grad = parent.grad + g
                            
    # Operations
    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(np.full(self.shape, other, dtype=np.float32), device=self.device)
        return Add.apply(self, other)
        
    def __sub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(np.full(self.shape, other, dtype=np.float32), device=self.device)
        return Sub.apply(self, other)
        
    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(np.full(self.shape, other, dtype=np.float32), device=self.device)
        return Mul.apply(self, other)
        
    def __matmul__(self, other):
        return Matmul.apply(self, other)
        
    def __repr__(self):
        return f"Tensor({self.to_list()}, shape={self.shape}, device={'CUDA' if self.device == Device.CUDA else 'CPU'}, requires_grad={self.requires_grad})"
