from dataloader.csvData import batch_loader
from dataloader.csvData import z_normalize
from grad_engine.pd_to_tensor import pd_to_tensor
from layers.dense import linear
from grad_engine.value import sigmoid
from loss.binary_cross_entropy import binary_cross_entropy
from optims.sgd import sgd
import pandas as pd

df = pd.read_csv("datasets/breast_cancer.csv", skiprows=1, header=None)
# 1st column irrelevant (ID NO.)
df = df.drop(columns=[0])
# Map labels: 'M' → 1, 'B' → 0
df[1] = df[1].map({'M': 1, 'B': 0})

X = df.drop(columns=[1]).values 
X,mean,std=z_normalize(X)
y = df[1].values

print(y.shape)
#logistic reg
class logistic:
    def __init__(self):
        self.linear=linear(30,1,sigmoid)

    def __call__(self,x):
        o=self.linear(x)
        return o

model=logistic()  

#testing to see the loss converges for now
for xb,yb in batch_loader(X,y,batch_size=16):
    xt= pd_to_tensor(xb)

    o=model(xt)
    loss=binary_cross_entropy(o,yb)
    print(loss)

    loss.backward()
    sgd(loss,0.0001)
