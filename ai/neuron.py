import math
import numpy as np
import matplotlib.pyplot as plt
import random


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def der_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


class neuron:
    def __init__(self):
        self.weights = []
        self.bias = 1
        self.output = None

    def randomize(self, input_count, min_val=-1, max_val=1, weights=None, bias=None):
        self.weights = []
        for i in range(input_count):
            if weights is None: self.weights.append(random.uniform(min_val, max_val))
            else: self.weights.append(weights[i]+random.uniform(min_val, max_val))
        if bias is None: self.bias = random.uniform(min_val, max_val)
        else: self.bias = bias+random.uniform(min_val, max_val)

    def calc(self, inputs):
        self.output = 0
        for i in range(len(inputs)):
            self.output += inputs[i] * self.weights[i]
        return self.output


x = 1 / (1 + math.exp(-1))
