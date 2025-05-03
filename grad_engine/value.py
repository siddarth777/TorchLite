import math

#individual number class which retains its grad and how its made
class value:
    def __init__(self,val,hist=(),op='',trainable=False):
        self.val=val
        self.grad=0
        #values used to create self
        self.hist=set(hist)
        #operation used to create self
        self.op=op
        #the process to get derivate wrt hist from derivate wrt self
        self._backward= lambda:None
        self.trainable=trainable
    
    #Data shown when printing the object
    def __repr__(self):
        return f'value: {self.val} with gradient: {self.grad}'

    #Addition
    '''
    c=a+b
    dl/dc = k (known)
    dc/da= 1  (partial derivate wrt a)

    dl/da =  dl/dc* da/dc
    dl/da = k
    '''
    def __add__(self, other):
        other = other if isinstance(other, value) else value(other)
        out = value(self.val + other.val, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out
    
    """
    += used instead of = in _backward to allow gradients to stack

    ex:
    c=a+b
    d=a*b
    e=c+d

    during e.backward()
    a will first get assigned 1 from c (since e=c+d and c=a+b)
    and then 1 will be overwritten by b*1 from d (since e=c+d and d=a*b)
    however we know that the real answer is 1+b
    """
    
    #Subtraction
    #same as addition but with a negative relation
    def __sub__(self,other):
        other = other if isinstance(other, value) else value(other)
        out = value(self.val + other.val, (self, other), '+')

        def _backward():
            self.grad -= out.grad
            other.grad -= out.grad

        out._backward = _backward
        return out
    
    #Multiplication
    '''
    c=a*b
    dl/dc = k  (known)
    dc/da= 0 + b (partial derivate wrt a)

    dl/da = dl/dc*dc/da
    dl/da = k*b
    '''
    def __mul__(self,other):
        other = other if isinstance(other, value) else value(other)
        out = value(self.val*other.val,(self,other),'*')

        def _backward():
            self.grad+=other.val*out.grad
            other.grad+=self.val*out.grad

        out._backward= _backward
        return out
    
    #Exponentiation
    '''
    c=a^k (k is constant)
    dc/da = k*a^(k-1) (partial derivate wrt a)

    dl/da= dl/dc* k*a^(k-1)
    '''
    def pow(self,x):
        out=value(self.val**x,(self,),'^')

        def _backward():
            self.grad+=out.grad*(x)*(self.val**(x-1))
        
        out._backward=_backward
        return out

    #Backprop
    '''
    TOPOLOGICAL ORDER REQUIRED SINCE THE COMPUTATION IS UNIDIRECTIONAL
    ex:
    a+b=c
    m+n=l
    x=c*l
    y=b+n
    z=x*y

    starting from z
    gradient will flow into x and y
    x will flow into c and l
    y will flow into b and n
    similarly 
    c will flow into a and b
    l will flow into m and n

    creating the order
    z->x->y->c->l->b->n->a->m (the ._backwards of a,b,m,n will do nothing)
    this is 1 possible order (you can see that we can swap some of these locations and it will still work)
    however we can also see that doing c._backward before x._backward will give wrong wrong (since c.grad depends on x.grad )
    so we must follow a reversed topological sequence to prevent mis-calcuations 
    '''
    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.hist:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1
        for v in reversed(topo):
            v._backward()

#ReLU
'''
    f(x)=max(0,x)
    f'(x)= {
            0   x<0
            1   x>0
            }

    c = f(a)
    dl/dc = k (known)
    dc/da = f'(a) (partial derivate wrt a)

    dl/da= k*f'(a)
'''
def relu(self):
    out = value(0 if self.val < 0 else self.val, (self,), 'ReLU')

    def _backward():
        self.grad += (out.val > 0) * out.grad

    out._backward = _backward

    return out

#Sigmoid
'''
    f(x)=1/(1+e^-x)
    f'(x)= (0*(1 + e^-x) - 1*(0 - e^-x) ) / (1 + e^-x)^2
    re-arranging
    f'(x)= 1/(1+e^-x) * (1 - 1/(1+e^-x))
    f'(x)= f(x)*(1-f(x))

    c = f(a)
    dl/dc = k (known)
    dc/da = f'(a) (partial derivate wrt a)

    dl/da= k*f(a)*(1-f(a))
'''
def sigmoid(self):
    out=value(1/(1+math.exp(self.val*-1)),(self,),'sigmoid')

    def _backward():
        self.grad+=out.val*(1-out.val)*out.grad
        
    out._backward = _backward

    return out