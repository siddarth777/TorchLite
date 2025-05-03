from grad_engine.tensor import tensor
from grad_engine.value import value
def dot(t1:tensor,t2:tensor):
    if(t1.dim!=t2.dim):
        raise ValueError(f"Tensor dimension mismatch: {t1.dim} vs {t2.dim}")
    out=tensor([1],root=False)
    for x in range(t1.len):
        out.array[0]+=t1.array[x]*t2.array[x]
    return out