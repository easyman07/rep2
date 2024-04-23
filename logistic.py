import numpy as np
import matplotlib as mat
import math

# Entering dataset
n = 5

x1 = [9.9, 10.7, 0.34, 7.4, 1.6]
x2 = [8.4, 10.9, 0.2, 8.32, 2.56]
y = [1, 1, 0, 1, 0]

# print(x1)
# print(x2)
# print(y)

b0 = 0 
b1 = 0
b2 = 0

s = 0.8

p = []
pc = []


for i in range(n) : 
    p.append(1 / (1 + np.exp(-(b0 + b1 * x1[i] + b2 * x2[i]))))
    # print(p[i])

    b0 = b0 + s * (y[i] - p[i]) * p[i] * (y[i] - p[i]) * 1
    b1 = b1 + s * (y[i] - p[i]) * p[i] * (y[i] - p[i]) * x1[i]
    b2 = b2 + s * (y[i] - p[i]) * p[i] * (y[i] - p[i]) * x2[i]

    if(p[i] < s):
        pc.append(0)
        # print(0)
    else:
        pc.append(1)
        # print(1)


print("P is: ")
print(p)
print("PC: ")
print(pc)