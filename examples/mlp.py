import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the root directory to path to import torchlite
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torchlite
from torchlite.tensor import Function, Tensor

class ReLU(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        arr = a.numpy()
        out = np.maximum(0, arr)
        return Tensor(out, device=a.device, requires_grad=False)
        
    @staticmethod
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        arr = a.numpy()
        grad = grad_output.numpy()
        grad[arr <= 0] = 0
        return Tensor(grad, device=grad_output.device)

class AddBias(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        out = a.numpy() + b.numpy()
        return Tensor(out, device=a.device, requires_grad=False)
        
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_out_np = grad_output.numpy()
        grad_a = Tensor(grad_out_np, device=grad_output.device)
        grad_b = Tensor(np.sum(grad_out_np, axis=0, keepdims=True), device=grad_output.device)
        return grad_a, grad_b

class MSELoss(Function):
    @staticmethod
    def forward(ctx, pred, target):
        ctx.save_for_backward(pred, target)
        diff = pred.numpy() - target.numpy()
        loss = float(np.mean(diff ** 2))
        return Tensor(loss, device=pred.device, requires_grad=False)
        
    @staticmethod
    def backward(ctx, grad_output):
        pred, target = ctx.saved_tensors
        diff = pred.numpy() - target.numpy()
        N = pred.shape[0]
        grad = (2.0 / N) * diff * grad_output.numpy()
        return Tensor(grad, device=grad_output.device), Tensor(-grad, device=grad_output.device)

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialization
        np.random.seed(42)
        W1_np = np.random.randn(input_size, hidden_size).astype(np.float32) * np.sqrt(2.0 / input_size)
        b1_np = np.zeros((1, hidden_size), dtype=np.float32)
        W2_np = np.random.randn(hidden_size, output_size).astype(np.float32) * np.sqrt(2.0 / hidden_size)
        b2_np = np.zeros((1, output_size), dtype=np.float32)
        
        self.W1 = Tensor(W1_np, requires_grad=True)
        self.b1 = Tensor(b1_np, requires_grad=True)
        self.W2 = Tensor(W2_np, requires_grad=True)
        self.b2 = Tensor(b2_np, requires_grad=True)
        
    def parameters(self):
        return [self.W1, self.b1, self.W2, self.b2]
        
    def forward(self, x):
        h = x @ self.W1
        h = AddBias.apply(h, self.b1)
        h = ReLU.apply(h)
        out = h @ self.W2
        out = AddBias.apply(out, self.b2)
        return out

def generate_moons(n_samples=200, noise=0.1):
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out
    
    outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out))
    outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out))
    inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, n_samples_in))
    inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, n_samples_in)) - 0.5
    
    X = np.vstack([np.append(outer_circ_x, inner_circ_x),
                   np.append(outer_circ_y, inner_circ_y)]).T
    y = np.hstack([np.zeros(n_samples_out, dtype=np.float32),
                   np.ones(n_samples_in, dtype=np.float32)])
                   
    X += np.random.randn(*X.shape) * noise
    return X.astype(np.float32), y.reshape(-1, 1).astype(np.float32)

def train_mlp():
    np.random.seed(42)
    X_np, y_np = generate_moons(n_samples=300, noise=0.15)
    
    X = Tensor(X_np)
    y = Tensor(y_np)
    
    # Increase hidden size for more complex decision boundary
    model = MLP(2, 16, 1)
    
    lr = 0.5
    epochs = 2000
    losses = []
    
    print("Training on Moons dataset...")
    for epoch in range(epochs):
        pred = model.forward(X)
        loss = MSELoss.apply(pred, y)
        losses.append(float(loss.numpy()[0]))
        
        loss.backward()
        
        for param in model.parameters():
            if param.grad is not None:
                new_data = param.numpy() - lr * param.grad.numpy()
                temp_tensor = Tensor(new_data, device=param.device)
                param._tensor = temp_tensor._tensor
                param.grad = None
                
        if (epoch + 1) % 200 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {float(loss.numpy()[0]):.4f}')
            
    # Save loss plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title("MLP Training Loss on Moons")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    
    # Plot Decision Boundary
    plt.subplot(1, 2, 2)
    x_min, x_max = X_np[:, 0].min() - 0.5, X_np[:, 0].max() + 0.5
    y_min, y_max = X_np[:, 1].min() - 0.5, X_np[:, 1].max() + 0.5
    h = 0.05
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
    grid_tensor = Tensor(grid)
    Z = model.forward(grid_tensor).numpy()
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z > 0.5, alpha=0.8, cmap=plt.cm.Spectral)
    plt.scatter(X_np[:, 0], X_np[:, 1], c=y_np.reshape(-1), cmap=plt.cm.Spectral, edgecolors='k')
    plt.title("Decision Boundary")
    
    save_path = os.path.join(os.path.dirname(__file__), "moons_convergence.png")
    plt.savefig(save_path)
    print(f"Convergence plot saved to {save_path}")

if __name__ == "__main__":
    train_mlp()
