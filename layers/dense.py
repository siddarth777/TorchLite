from grad_engine.tensor import tensor
from grad_engine.matmul import matmul2d
'''
W stores weights for each node as a column vector
thus there are s_out columns each for a node

this allows x to be a row vectors

works when x is X and has multiple inputs (row wise entry)
each output row will be for a different input

X: batchsize,s_in
W: s_in,s_out

X*W= batchsize,s_out
b: s_out (applied to each row of X*W)
'''
class linear:
    def __init__(self,s_in,s_out,activation=lambda:None):
        self.W=tensor([s_in,s_out],trainable=True)
        self.b=tensor([s_out],trainable=True)
        self.activation=activation

    def __call__(self, X:tensor):
        Z=matmul2d(X,self.W)
        for i in range(self.W.dim[1]):
            for batch in range (X.dim[0]):
                Z.array[batch*self.W.dim[1]+i]+=self.b.array[i]

        #testing code
        g=tensor(Z.dim,root=False)
        for i in range(Z.len):
            g.array[i]= self.activation(Z.array[i])
        return g