import numpy as np

class Hebb:
    def __init__(self):
        pass

    def HebbAND(self):
        print("\n")
        # For AND Gate Iteration
        x1 = [-1, -1, 1, 1] # Bipolar values
        x2 = [-1, 1, -1, 1] # Bipolar values
        b = [1, 1, 1, 1]
        y = [-1, 1, 1, 1]  # target

        w1 = 0
        w2 = 0
        bias = 0  # Use a different variable name for the bias term
        print(x1, x2, b, y)
        for i in range(0, 4):
            w1n = w1 + x1[i] * y[i]
            w2n = w2 + x2[i] * y[i]
            bias_n = bias + y[i]
            print("Weights and bias after iteration " + str(i) + " :")
            print("W1: " + str(w1n))
            print("W2: " + str(w2n))
            print("Bias: " + str(bias_n))

            w1 = w1n
            w2 = w2n
            bias = bias_n

        print("\n")
        print("Final Weights:")
        print(w1, w2, bias)

# Create an instance of the Hebb class
hebb_instance = Hebb()
hebb_instance.HebbAND()
