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


def FourNetwork(X, W1, W2, W3, W4, b1, b2, b3, b4, y, test, learning_rate):
    Z1, acache1 = AffineForward(X, W1, b1)
    A1, rcache1 = ReLUForward(Z1)
    Z2, acache2 = AffineForward(A1, W2, b2)
    A2, rcache2 = ReLUForward(Z2)
    Z3, acache3 = AffineForward(A2, W3, b3)
    A3, rcache3 = ReLUForward(Z3)
    F, acache4 = AffineForward(A3, W4, b4)
    if test:
        return np.argmax(F, axis=1)
    loss, dF = CrossEntropy(F, y)
    dA3, dW4, db4 = AffineBackwards(dF, acache4)
    dZ3 = ReLUBackward(dA3, rcache3)
    dA2, dW3, db3 = AffineBackwards(dZ3, acache3)
    dZ2 = ReLUBackward(dA2, rcache2)
    dA1, dW2, db2 = AffineBackwards(dZ2, acache2)
    dZ1 = ReLUBackward(dA1, rcache1)
    dX, dW1, db1 = AffineBackwards(dZ1, acache1)
    W1 -= learning_rate*dW1
    W2 -= learning_rate*dW2
    W3 -= learning_rate*dW3
    W4 -= learning_rate*dW4
    return loss


def MiniBatchCD(data, epoch, batch_size, weight_scale, learning_rate):
    # Init W1, W2, W3, W4, b1, b2, b3, b4
    W1 = (2*np.random.rand(5, 256) - 1)*weight_scale
    b1 = np.zeros((256,))
    W2 = (2*np.random.rand(256, 256) - 1)*weight_scale
    b2 = np.zeros((256,))
    W3 = (2*np.random.rand(256, 256) - 1)*weight_scale
    b3 = np.zeros((256,))
    W4 = (2*np.random.rand(256, 3) - 1)*weight_scale
    b4 = np.zeros((3,))
    for e in range(epoch):
        print("epoch", e+1)
        np.random.shuffle(data)
        for i in range(data.shape[0]//batch_size):
            # X, y = batch of features and targets from data
            X = data[i*batch_size:(i+1)*batch_size, 0:5]
            y = data[i*batch_size:(i+1)*batch_size, 5]
            loss = FourNetwork(X, W1, W2, W3, W4, b1, b2, b3, b4, y, False, learning_rate)
        X = data[:, 0:5]
        y = data[:, 5]
        result = FourNetwork(X, W1, W2, W3, W4, b1, b2, b3, b4, y, True, learning_rate)
        print(np.sum(np.equal(result, data[:, 5]))/len(result))


expert_policy = np.genfromtxt("expert_policy.txt", delimiter=" ")
expert_policy[:, 0:5] = ((expert_policy - expert_policy.mean(axis=0))/expert_policy.std(axis=0))[:, 0:5]
MiniBatchCD(expert_policy, 300, 250, 0.5, 0.1)
print("done")
