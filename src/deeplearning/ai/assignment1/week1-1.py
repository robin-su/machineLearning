# -*- coding:utf-8 -*-
import math
import numpy as np
import time

test = "Hello World"
print("test:" + test)

def basic_sigmoid(x):
    s = 1/(1 + math.exp(-x))
    return s

print(basic_sigmoid(3))

x = np.array([1,2,3])
print(np.exp(x))

print(x + 3)

def sigmoid(x):
    s = 1/(1 + np.exp(-x))
    return s

print(sigmoid(x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    ds = s * (1-s)
    return ds

print("sigmoid_derivative(x) = " + str(sigmoid_derivative(x)))

print("--------------------- shape/reshape -------------------------")

def image2vector(image):
    v = image.reshape(image.shape[0] * image.shape[1] * image.shape[2],1)
    return v

image = np.array([[[0.67826139,0.29380381],[0.90714982,0.52835647],[0.4215251,0.45017551]],
                  [[0.92814219,0.96677647],[0.85304703,0.52351845],[0.19981397,0.27417313]],
                  [[0.60659855,0.00533165],[0.10820313,0.49978937],[0.34144279,0.94630077]]])

print(image.shape)
print ("image2vector(image) = " + str(image2vector(image)))
print(image2vector(image).shape)

print("--------------------- normalizeRows -------------------------")

def normalizeRows(x):
    x_norm = np.linalg.norm(x,axis=1,keepdims=True) # 按行标准化
    x = x/x_norm
    return x

x = np.array([[0,3,4],
              [1,6,4]])

print("normalizeRows(x) = " + str(normalizeRows(x)))

print("--------------------- softmax -------------------------")

def softmax(x):

    x_exp = np.exp(x)
    x_sum = np.sum(x_exp,axis=1,keepdims=True)

    s = x_exp/x_sum
    return s

x = np.array([[9,2,5,0,0],[7,5,0,0,0]])
print("softmax(x) = " + str(softmax(x)))

print("--------------------- CLASSIC DOT PRODUCT OF VECTORS IMPLEMENTATION  -------------------------")
x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]

tic = time.process_time()
dot = 0
for i in range(len(x1)):
    dot += x1[i] * x2[i]

toc = time.process_time()
print("dot = " + str(dot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

tic1 = time.process_time()
outer = np.zeros((len(x1),len(x2)))

for i in range(len(x1)):
    for j in range(len(x2)):
        outer[i,j] = x1[i] * x2[j]

toc2 = time.process_time()
print ("outer = " + str(outer) + "\n ----- Computation time = " + str(1000*(toc2 - tic1)) + "ms")

tic3 = time.process_time()
mul = np.zeros(len(x1))
for i in range(len(x1)):
    mul[i] = x1[i] * x2[i]
toc3 = time.process_time()

print ("elementwise multiplication = " + str(mul) + "\n ----- Computation time = " + str(1000*(toc3 - tic3)) + "ms")

w = np.random.rand(3,len(x1))
print("w = " + str(w))
tic4 = time.process_time()
gdot = np.zeros(w.shape[0])

for i in range(w.shape[0]):
    for j in range(w.shape[1]):
        gdot[i] += w[i,j] * x1[j]

toc = time.process_time()
print ("gdot = " + str(gdot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

print("--------------------- VECTORIZED DOT PRODUCT OF VECTORS -------------------------")
tic = time.process_time()
dot = np.dot(x1,x2) # 两个向量的点乘
toc = time.process_time()
print("outer = " + str(outer) + "\n ------- Computation time = " + str(1000*(toc - tic)) + "ms")

tic = time.process_time()
outer = np.outer(x1,x2)  # 将乘积放入到矩阵对应的位置
toc = time.process_time()
print("outer = " + str(outer) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

tic = time.process_time()
mul = np.multiply(x1,x2) # 对应的位置相乘并复制给对应的位置
toc = time.process_time()
print("elementwise multiplication = " + str(mul) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

tic = time.process_time()
dot = np.dot(w,x1)
toc = time.process_time()
print ("gdot = " + str(dot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

print("--------------------- L2 and L1 loss functions -------------------------")
def L1(yhat,y):
    loss = np.sum(np.abs(y - yhat))
    return loss

yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L1 = " + str(L1(yhat,y)))

def L2(yhat,y):
    loss = np.dot((y - yhat),(y - yhat))
    return loss

yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L2 = " + str(L2(yhat,y)))