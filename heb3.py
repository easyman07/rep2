import numpy as np

class Hebb:
    def __init__(self):
        pass

    def hebb_learning(self, x1, x2, y):
        w1 = 0
        w2 = 0
        bias = 0
        print("\n")
        print("Initial Weights and Bias:")
        print("W1:", w1)
        print("W2:", w2)
        print("Bias:", bias)

        for i in range(0, 4):
            w1n = w1 + x1[i] * y[i]
            w2n = w2 + x2[i] * y[i]
            bias_n = bias + y[i]
            print("\nWeights and bias after iteration " + str(i) + ":")
            print("W1:", w1n)
            print("W2:", w2n)
            print("Bias:", bias_n)

            w1 = w1n
            w2 = w2n
            bias = bias_n

        print("\nFinal Weights and Bias:")
        print("W1:", w1)
        print("W2:", w2)
        print("Bias:", bias)

    def HebbNAND(self):
        print("\nHebbian Learning for NAND Gate:")
        x1 = [-1, -1, 1, 1]  # Bipolar values
        x2 = [-1, 1, -1, 1]  # Bipolar values
        b = [1, 1, 1, 1]
        y = [1, 1, 1, -1]  # target
        self.hebb_learning(x1, x2, y)

    def HebbAND(self):
        print("\nHebbian Learning for AND Gate:")
        x1 = [-1, -1, 1, 1]  # Bipolar values
        x2 = [-1, 1, -1, 1]  # Bipolar values
        b = [1, 1, 1, 1]
        y = [-1, 1, 1, 1]  # target
        self.hebb_learning(x1, x2, y)

    def HebbOR(self):
        print("\nHebbian Learning for OR Gate:")
        x1 = [-1, -1, 1, 1]  # Bipolar values
        x2 = [-1, 1, -1, 1]  # Bipolar values
        b = [1, 1, 1, 1]
        y = [-1, 1, 1, 1]  # target
        self.hebb_learning(x1, x2, y)

    def HebbXOR(self):
        print("\nHebbian Learning for XOR Gate:")
        x1 = [-1, -1, 1, 1]  # Bipolar values
        x2 = [-1, 1, -1, 1]  # Bipolar values
        b = [1, 1, 1, 1]
        y = [-1, 1, 1, -1]  # target
        self.hebb_learning(x1, x2, y)

# Create an instance of the Hebb class
hebb_instance = Hebb()
hebb_instance.HebbNAND()
hebb_instance.HebbAND()
hebb_instance.HebbOR()
hebb_instance.HebbXOR()
