import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


class Cache(object):
    def __init__(self, A=None, W=None, Z=None):
        self.A = np.copy(A)
        self.W = np.copy(W)
        self.Z = np.copy(Z)


# Writes learned policy to file
# format, "our_policy[datetime].txt"
def store_policy(W1, W2, W3, W4, b1, b2, b3, b4):
    with open("our_policy" + datetime.now().strftime('%Y%m%d%H%M%S') + ".txt", 'w') as f:
        f.write('W1')
        f.write('\n')
        W1.tofile(f, sep=" ")
        f.write('\n')
        f.write('\n')
        f.write('b1')
        f.write('\n')
        b1.tofile(f, sep=" ")
        f.write('\n')
        f.write('\n')
        f.write('W2')
        f.write('\n')
        W2.tofile(f, sep=" ")
        f.write('\n')
        f.write('\n')
        f.write('b2')
        f.write('\n')
        b2.tofile(f, sep=" ")
        f.write('\n')
        f.write('\n')
        f.write('W3')
        f.write('\n')
        W3.tofile(f, sep=" ")
        f.write('\n')
        f.write('\n')
        f.write('b3')
        f.write('\n')
        b3.tofile(f, sep=" ")
        f.write('\n')
        f.write('\n')
        f.write('W4')
        f.write('\n')
        W4.tofile(f, sep=" ")
        f.write('\n')
        f.write('\n')
        f.write('b4')
        f.write('\n')
        b4.tofile(f, sep=" ")
        f.write('\n')


# Prints confusion matrix and Misclassification Error
def printConfusionMisclassification(data, policy):
    W1 = np.genfromtxt(policy, max_rows=1, skip_header=1, delimiter=" ").reshape((5, 256))
    b1 = np.genfromtxt(policy, max_rows=1, skip_header=4, delimiter=" ")
    W2 = np.genfromtxt(policy, max_rows=1, skip_header=7, delimiter=" ").reshape((256, 256))
    b2 = np.genfromtxt(policy, max_rows=1, skip_header=10, delimiter=" ")
    W3 = np.genfromtxt(policy, max_rows=1, skip_header=13, delimiter=" ").reshape((256, 256))
    b3 = np.genfromtxt(policy, max_rows=1, skip_header=16, delimiter=" ")
    W4 = np.genfromtxt(policy, max_rows=1, skip_header=19, delimiter=" ").reshape((256, 3))
    b4 = np.genfromtxt(policy, max_rows=1, skip_header=22, delimiter=" ")

    classifications = FourNetwork(data[:,0:5], W1, W2, W3, W4, b1, b2, b3, b4, data[:,5], True, 0)
    confusionMatrix = np.zeros((3, 3))
    correct = 0
    for i in range(len(classifications)):
        if classifications[i] == data[i, 5]:
            correct += 1
        confusionMatrix[int(data[i, 5])][int(classifications[i])] += 1

    for r in range(3):
        num_labels = np.sum(confusionMatrix[r])
        for c in range(3):
            confusionMatrix[r][c] /= num_labels

    print("Misclassification Error:", 1.0-(correct/10000))
    print()
    print(confusionMatrix)


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


def MiniBatchCD(data, epoch, batch_size, weight_scale):
    # Init W1, W2, W3, W4, b1, b2, b3, b4
    W1 = (2*np.random.rand(5, 256) - 1)*weight_scale
    b1 = np.zeros((256,))
    W2 = (2*np.random.rand(256, 256) - 1)*weight_scale
    b2 = np.zeros((256,))
    W3 = (2*np.random.rand(256, 256) - 1)*weight_scale
    b3 = np.zeros((256,))
    W4 = (2*np.random.rand(256, 3) - 1)*weight_scale
    b4 = np.zeros((3,))
    epochs = [e+1 for e in range(epoch)]
    losses = []
    accuracies = []
    for e in range(epoch):
        print("epoch", e+1)
        np.random.shuffle(data)
        total_loss = 0
        batch_count = 0
        for i in range(data.shape[0]//batch_size):
            # X, y = batch of features and targets from data
            X = data[i*batch_size:(i+1)*batch_size, 0:5]
            y = data[i*batch_size:(i+1)*batch_size, 5]
            total_loss += FourNetwork(X, W1, W2, W3, W4, b1, b2, b3, b4, y, False, -1.983968e-04*e+0.1001983968)
            batch_count += 1
        losses.append(total_loss/batch_count)
        print("average loss", total_loss/batch_count)
        X = data[:, 0:5]
        y = data[:, 5]
        result = FourNetwork(X, W1, W2, W3, W4, b1, b2, b3, b4, y, True, 0.0)
        accuracy = np.sum(np.equal(result, data[:, 5]))/len(result)
        print("accuracy", accuracy)
        accuracies.append(accuracy)

    plt.semilogy(epochs, losses)
    plt.title("Loss vs. Training Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("loss_v_epochs_semilogy" + datetime.now().strftime('%Y%m%d%H%M%S') + ".png")
    plt.close()
    plt.figure()
    plt.plot(epochs, accuracies)
    plt.title("Accuracy vs. Training Epochs")
    plt.ylabel("Epoch")
    plt.xlabel("Accuracy")
    plt.savefig("accuracy_v_epochs" + datetime.now().strftime('%Y%m%d%H%M%S') + ".png")
    return W1, W2, W3, W4, b1, b2, b3, b4


expert_policy = np.genfromtxt("expert_policy.txt", delimiter=" ")
expert_policy[:, 0:5] = ((expert_policy - expert_policy.mean(axis=0))/expert_policy.std(axis=0))[:, 0:5]
W1, W2, W3, W4, b1, b2, b3, b4 = MiniBatchCD(expert_policy, 500, 250, 0.5)
store_policy(W1, W2, W3, W4, b1, b2, b3, b4)

# printConfusionMisclassification(expert_policy, "our_policy20180430105221.txt")
# Misclassification Error: 0.010199999999999987
#
# [[ 0.98710258  0.00719856  0.00569886]
#  [ 0.00554844  0.98804951  0.00640205]
#  [ 0.00462642  0.00254453  0.99282905]]
