import numpy as np


class Cache(object):
    def __init__(self, A=None, W=None, Z=None):
        self.A = np.copy(A)
        self.W = np.copy(W)
        self.Z = np.copy(Z)


# Description: Computes affine transformation Z = AW + b.
# Inputs: Incoming batch of data A (2d array of size n x d), layer weight W (2d array of size d x d'), bias b (1d array of size d')
# Returns: affine output Z (2d array of size nxd'), cache object
# This function basically takes n training examples with d input features and creates a new dataset of n training examples with d' output features. A, W, and Z are 2d arrays and b is a 1d array. The number of output features d', also referred to as the number of layer units, is determined by you except for the last (output) layer, which must be 3. The function computes the following affine transformation.
def AffineForward(A, W, b):
    return A@W + b, Cache(A=A, W=W)


# Computes the gradients of loss L with respect to forward inputs A, W, b.
# Inputs: Gradient dZ, cache object from forward operation
# Returns: dA, dW, db
# This function computes the gradients of A, W, and b with respect to loss. It is important to note that for any variable V and it's loss gradient dV have the same dimensions. Using the chain rule from calculus, the gradients dA, dW, and db are
def AffineBackwards(dZ, cache):
    dA = dZ@cache.W.T
    dW = cache.A.T@dZ
    db = dZ.sum(axis=0)
    return dA, dW, db


# Computes elementwise ReLU of Z.
# Inputs: Batch Z (2d array of size n x d')
# Returns: ReLU output A (2d array of size n x d'), cache object
# The ReLU function takes the elements of some 2d array A and zeros out the values that are negative. Thus Aij = Zij if Zij > 0, otherwise Aij = 0. The cache object stores the value of Z since Z is needed for the ReLU backwards computation.
def ReLUForward(Z):
    return np.maximum(Z, 0.0), Cache(Z=Z)


# Computes gradient of Z with respect to loss
# Inputs: Gradient dZ, cache object from forward
# Returns: gradient of Z with respect to loss L
# If Zij was zeroed out during the ReLU forward computation, then the gradient dZij must be 0 as well. This makes intuitive sense because if Zij was zeroed out, then Zij does not contribute to the loss L so the gradient of the loss L with respect to Zij must be 0. If Zij was not zeroed out, then dZij = dAij.
def ReLUBackward(dA, cache):
    return np.where(np.less_equal(cache.Z, 0.0), 0.0, dA)


# Description: Computes the loss function L and the gradients of the loss with respect to the scores F.
# Inputs: logits and target classes
# Returns: loss L and gradients dlogits.
def CrossEntropy(F, y):
    outer_sum = 0
    for i in range(F.shape[0]):
        fiyi = F[i, int(y[i])]
        inner_sum = 0
        for k in range(F.shape[1]):
            inner_sum += np.exp(F[i, k])
        outer_sum += fiyi - np.log(inner_sum)
    L = -(1/F.shape[0])*outer_sum

    dF = np.zeros(F.shape)
    for i in range(F.shape[0]):
        for j in range(F.shape[1]):
            val = 1 if j == y[i] else 0
            dF[i, j] = -(1/F.shape[0])*(val - np.exp(F[i, j])/np.sum(np.exp(F[i])))

    return L, dF


a = np.genfromtxt("check_affine.txt", max_rows=10, skip_header=1, delimiter=" ")
w = np.genfromtxt("check_affine.txt", max_rows=8, skip_header=13, delimiter=" ")
B = np.genfromtxt("check_affine.txt", max_rows=1, skip_header=23, delimiter=" ")
z, cache = AffineForward(a, w, B)

dz = np.genfromtxt("check_affine.txt", max_rows=10, skip_header=38, delimiter=" ")
dA, dW, db = AffineBackwards(dz, cache)
print(dA)
print(dW)
print(db)

z = np.genfromtxt("check_relu.txt", max_rows=10, skip_header=1, delimiter=" ")
A, cache = ReLUForward(z)
print(A)
print(cache.Z)

da = np.genfromtxt("check_relu.txt", max_rows=10, skip_header=25, delimiter=" ")
dz = ReLUBackward(da, cache)
print(dz)

f = np.genfromtxt("check_cross_entropy.txt", max_rows=10, skip_header=1, delimiter=" ")
Y = np.genfromtxt("check_cross_entropy.txt", max_rows=1, skip_header=13, delimiter=" ", dtype=int)
L, dF = CrossEntropy(f, Y)
print(L)
print(dF)
