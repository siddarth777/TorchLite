from grad_engine.value import value

# Standard Gradient Descent
def sgd(loss: value, alpha: float):
    visited = set()

    def backward(v: value):
        if v in visited:
            return
        visited.add(v)
        
        for child in v.hist:
            backward(child)
        
        if v.trainable:
            v.val -= alpha * v.grad
        v.grad = 0.0

    backward(loss)
