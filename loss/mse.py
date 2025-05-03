from grad_engine.tensor import tensor
from grad_engine.value import value
#mean squared error
'''
yhat: batchsize,ans_size (Matrix)
y : ans_size (vector)

mse = 1/batchsize * 1/ans_size (yhat-y)^2
'''
def mse(yhat:tensor,y):
    loss = value(0)
    batch_size,ans_size=yhat.dim
    for batch in range(batch_size):
        for i in range(ans_size):
            loss+=(yhat.array[batch*ans_size+i]-y[i]).pow(2)
    
    return loss*(1/(2*batch_size*ans_size))
    