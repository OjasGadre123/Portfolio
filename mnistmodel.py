
import numpy as np 
from graphviz import Digraph

# graph Node
class Tensor:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = np.zeros((data.shape))
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label
    
    def zero_grad(self):
        self.grad = np.zeros((self.data.shape))

    def __repr__(self):
        return f"Tensor(shape={self.data.shape})"

    def __add__(self, other):
        out = Tensor(np.add(self.data, other.data), (self, other), 'np.add')

        def _backward():
            self.grad  += out.grad
            other.grad += np.mean(out.grad, axis=0)
        
        out._backward = _backward
        return out 
    
    def __mul__(self, other):
        out = Tensor(np.matmul(self.data, other.data), (self, other), 'np.matmul')

        def _backward():
            self.grad += np.matmul(out.grad, np.transpose(other.data, (1, 0)))
            other.grad += np.mean(np.matmul(np.transpose(self.data, (0, 2, 1)), out.grad), axis=0)


        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,), 'relu')

        def _backward():
            self.grad += np.multiply((self.data > 0), out.grad)

        out._backward = _backward
        return out

    def softmax(self):
        exp_d = np.exp(self.data - np.max(self.data, axis=-1, keepdims=True))
        out_d = exp_d / np.sum(exp_d, axis=-1, keepdims=True)
        out   = Tensor(out_d, (self,), 'softmax')

        def _backward():
            self.grad += out.grad

        out._backward = _backward
        return out
    
    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)

        for node in reversed(topo):
            node._backward()
            

# Visualize graph
def trace(root):
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges

def draw_dot(root):
    dot = Digraph(format="svg", graph_attr={'rankdir':'LR'})

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))

        dot.node(name=uid, label="{%s | data %s | grad %s}" % (n._op, n.data.shape, n.grad.shape), shape='record')

        if n._op:
            dot.node(name = uid + n._op, label = n._op)
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot

# define loss function 
def cce_loss(y_true, y_pred, eps=1e-12):
    # avoid log(0)
    y_pred_d = np.clip(y_pred.data, eps, 1 - eps)
    
    # cross entropy for each sample
    loss = -np.sum(y_true.data * np.log(y_pred_d), axis=-1)
    
    # mean over batch
    return np.mean(loss)

import torch
from torchvision import datasets
from sklearn.model_selection import train_test_split

# Download training set
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=False
)

# Download test set
test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=False
)

print("LOADED DATA")

X_train    = train_dataset.data.numpy().reshape(60000, 1, 28*28) / 255.0
y_train    = train_dataset.targets.numpy()
y_train_oh = np.eye(10)[y_train].reshape(60000, 1, 10)

X_test  = test_dataset.data.numpy() / 255.0
X_test  = X_test.reshape(X_test.shape[0], 1, 28*28)
y_test  = test_dataset.data.numpy()


w1 = Tensor(np.random.normal(loc = 0, scale = np.sqrt(2/((28*28) + 256)), size=(28*28, 256)))
b1 = Tensor(np.zeros((1, 256)))

w2 = Tensor(np.random.normal(loc = 0, scale = np.sqrt(2/(256+ 10)), size=(256, 10)))
b2 = Tensor(np.zeros((1,10)))


epochs = 5
batch_size = 10

for epoch in range(epochs):
    loss = 0
    for i in range(0, 60000, batch_size):
        x = Tensor(X_train[i:i+batch_size])

        # forward model
        o1 = x * w1
        o2 = o1 + b1
        o3 = o2.relu()
        o4 = o3 * w2
        o5 = o4 + b2
        o6 = o5.softmax()
    
        if i == 0:
            graph = draw_dot(o6)
            graph.render("graph", cleanup=True)

        # find loss
        loss += cce_loss(y_train[i:i+batch_size], o6.data)

        # back propogation
        o6.grad = o6.data - y_train_oh[i:i+batch_size]

        o6.backward()

        # update weights and bias
        lr = 0.01

        w1.data -= lr * w1.grad
        b1.data -= lr * b1.grad
        w2.data -= lr * w2.grad
        b2.data -= lr * b2.grad

    # zero grad
    w1.zero_grad()
    b1.zero_grad()
    w2.zero_grad()
    b2.zero_grad()

    print(f"avg loss: {loss / 6000}")




correct = 0


for x in range(X_test.shape[0]):
    o1 = Tensor(X_test[x]) * w1
    o2 = o1 + b1 
    o3 = o2.relu()
    o4 = o3 * w2
    o5 = o4 + b2
    o6 = o5.softmax()
    
    if np.argmax(o6.data) == y_test[x]:
        correct += 1

print(f"accuracy: {100 * (correct / X_test.shape[0])}%")

