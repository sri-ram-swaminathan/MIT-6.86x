import numpy as np

def randomization(n):
    
    return np.random.random([n,1])

print("A random 3x1 matrix is",randomization(3))

def operations(h, w):
    
    A = np.random.random([h,w])
    B = np.random.random([h,w])
    s = A+B
    return A,B,s
    
print("2 random 2x3 matrices and their sum are",operations(2,3))

def norm(A, B):
    
    s=A+B
    return np.linalg.norm(s)

A=np.array([1,2])
B=np.array([3,4])
print("The norm of the sum of [1,2] and [3,4] is", norm(A,B))

def neural_network(inputs, weights):
    """
     Takes an input vector and runs it through a 1-layer neural network
     with a given weight matrix and returns the output.

     Arg:
       inputs - 2 x 1 NumPy array
       weights - 2 x 1 NumPy array
     Returns (in this order):
       out - a 1 x 1 NumPy array, representing the output of the neural network
    """
    weights_transpose=weights.transpose()
    z=np.matmul(weights_transpose,inputs)
    z=np.tanh(z)
    return z

def scalar_function(x, y):
    
    if x<=y:
        return x*y
    else:
        return x/y

print("f(x,y) for x=2 and y=3 is",scalar_function(2,3))

def vector_function(x, y):
    
    vfunc=np.vectorize(scalar_function)
    return vector_function(x,y)

print("f(x,y) for x=[1,2,3,4] and y=2 is",vector_function([1,2,3,4],2))
