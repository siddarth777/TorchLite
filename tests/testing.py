import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from grad_engine.tensor import tensor
from grad_engine.dotProd import dot
from layers.dense import linear
from grad_engine.value import relu
from loss.mse import mse
from optims.sgd import sgd

test_input=tensor([10,3])

l1= linear(3,2,relu)

o=l1(test_input)
y=[2,10]
loss= mse(o,y)

loss.backward()

print(l1.W)
print(l1.b)
print(loss)

sgd(loss,1)

print(l1.W)
print(l1.b)
print(loss)