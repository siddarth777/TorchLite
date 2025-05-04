from dataloader.csvData import CSVdataloader
from dataloader.csvData import batch_loader
from dataloader.csvData import z_normalize
from grad_engine.pd_to_tensor import pd_to_tensor
from layers.dense import linear
from grad_engine.value import relu
from loss.mse import mse
from optims.sgd import sgd

class MLP:
    #showcase example with 3 hidden layers of sizes 16 8 4 
    def __init__(self,input_size):
        self.l1=linear(input_size,16,relu)
        self.l2= linear(16,8,relu)
        self.l3= linear(8,4,relu)
        self.l4 = linear(4,1,relu)

    def __call__(self,x):
        t1=self.l1(x)
        t2=self.l2(t1)
        t3=self.l3(t2)
        o=self.l4(t3)

        return o

dataset_path = "datasets/boston.csv"
X_train, X_test, y_train, y_test = CSVdataloader(dataset_path)
X_train,mean,std = z_normalize(X_train)
y_train,_mean,_std = z_normalize(y_train)
model=MLP(14)

for xb, yb in batch_loader(X_train, y_train, batch_size=4):
    #training needs yb yo be 1 dimensional array-like
    yb=yb.reshape(-1)
    xt= pd_to_tensor(xb)
    
    o=model(xt)
    loss=mse(o,yb)
    print(loss)

    loss.backward()
    sgd(loss,0.0001)


