from grad_engine.tensor import tensor
from grad_engine.value import value
from grad_engine.value import log

def binary_cross_entropy(yhat:tensor,y):
    loss = value(0)
    batch_size,ans_size=yhat.dim

    for batch in range(batch_size):
        loss-=(log(yhat.array[batch])*y[batch] + log(value(1)-yhat.array[batch]))*(1-y[batch])
    
    return loss*(1/(batch_size))