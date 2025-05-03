import math
import random
from grad_engine.value import value
def initVal():
    t= random.uniform(-1,1)
    #normalized = math.exp(-1*t*t)
    return t

class tensor:
    '''
    A N dimensional tensor is technicaly just an really long array
    [3,28,28] tensor is 3*28*28 element long array
    where index x,y,z is just (x*28*28 + y*28 + z)th index in the 1d array
    '''
    def __init__(self,sizes,root=True,trainable=False):
        self.dim=sizes
        self.len=math.prod(sizes)
        self.array=[value(initVal(),trainable=trainable) for x in range(self.len)] if root else [value(0,trainable=trainable) for x in range(self.len)]
        self.trainable=trainable
        
    def __repr__(self):
        return '\n'.join(str(item) for item in self.array)+'\n'
    
    def __add__(self,other):
        if(self.dim==other.dim):
            out=tensor(self.dim,root=False)
            for x in range(self.len):
                out.array[x]=self.array[x]+other.array[x]
            return out
        else:
            raise ValueError(f"Tensor dimension mismatch: {self.dim} vs {other.dim}")
    def backward(self):
        for x in range(self.len):
            self.array[x].backward()

