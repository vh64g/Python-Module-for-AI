import numpy as np
import random


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def der_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


class neuron:
    def __init__(self):
        self.weights = None
        self.bias = 1
        self.output = None

    def randomize(self, input_count, min_val=-1, max_val=1):
        new_weights = []
        for i in range(input_count):
            new_weights.append(random.uniform(min_val, max_val))
        self.bias = random.uniform(min_val, max_val)
        self.weights = new_weights

    def calc(self, inputs):
        self.output = 0
        for i in range(len(inputs)):
            self.output += inputs[i] * self.weights[i]
        # self.output += self.bias
        # self.output = der_sigmoid(self.output)
        return self.output

