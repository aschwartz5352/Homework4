import numpy as np
from deep_learning import *


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
