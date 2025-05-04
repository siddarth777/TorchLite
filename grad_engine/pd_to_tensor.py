from grad_engine.tensor import tensor

def pd_to_tensor(x):
    dim = [ size for size in x.shape]
    xt = tensor(dim,root=False)
    #assuming 2d matrices
    for row in range(dim[0]):
        for column in range(dim[1]):
            xt.array[row*dim[1]+column].val=x[row][column]
    return xt
