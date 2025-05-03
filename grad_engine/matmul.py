from grad_engine.tensor import tensor
from grad_engine.value import value

def matmul2d(t1:tensor,t2:tensor):
    if(len(t1.dim)!=2 or len(t2.dim)!=2):
        raise ValueError("Not 2d matrices")
    if(t1.dim[1]!=t2.dim[0]):
        raise ValueError(f'Matmul not possible between tensors of size {t1.dim} and {t2.dim}')
    out=tensor([t1.dim[0],t2.dim[1]],root=False)
    for row in range(t1.dim[0]):
        for column in range(t2.dim[1]):
            for i in range(t1.dim[1]):
                out.array[row*out.dim[1]+column] += t1.array[row*t1.dim[1]+i]*t2.array[i*t2.dim[1]+column]
    return out
    