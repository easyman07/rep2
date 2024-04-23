# initalize the variables

import numpy as np


x1 = 0.35
x2 = 0.9 

w13 = 0.1
w14 = 0.4

w24 = 0.6
w23 = 0.8

w35 = 0.3
w45 = 0.9

y3 = 0
y4 = 0
y5 = 0
y = 0.5 # y is y target

def  forward (w1, x1, w2, x2, y):
    a = (w1 * x1) + (w2 * x2)
    # print(a)
    y = 1 / (1 + np.exp(-a))
    # print(y)
    return y


y3 = forward(w13, x1, w23, x2, y3)
print("y3: ",y3)
y4 = forward(w14, x1, w24, x2,y4)
print("y4: ",y4)
y5 = forward(w35, y3, w45, y4, y5)
print("y5: ",y5)

error = y - y5
print("Error:", error)

# calculate the error at each node
err3 = 0
err4 = 0

err5 = y5 * (1 - y5) * (y - y5)
print("error at 5 (output error)", err5)

def error(y, w, err):
    e = y * (1 - y) * (w * err)
    return e

err3 = error(y3, w35, err5)
print("error at 3: ", err3)

err4 = error(y4, w45, err5)
print("error at 4: ", err4)

# now calculate the new weights
# w45 = w45 + (1 * err5 * y4)
# print("new w45:" , w45)

def newWeights(wold, n, error, y):
    e = (n * error * y)
    wnew = wold + e
    return wnew


print("Updated weights are as follows: ")
w45 = newWeights(w45, 1, err5, y4)
print("w45: ", w45)

w35 = newWeights(w35, 1, err5, y3)
print("w35: ", w35)

w23 = newWeights(w23, 1, err3, x2)
print("w23: ", w23)

w24 = newWeights(w24, 1, err4, x2)
print("w24: ", w24)

w13 = newWeights(w13, 1, err3, x1)
print("w13: ", w13)

w14 = newWeights(w14, 1, err4, x1)
print("w14: ", w14)


# def main():
#     calculateForward()
#     error = y - y5
#     print("Error:", error)
    








