import math
import random


class neuron:
    def __init__(self):
        self.weights = []
        self.bias = 0
        self.output = 0

    def randomize(self, input_count, min_val=-1, max_val=1, weights=None, bias=None):
        self.weights = []
        for i in range(input_count):
            if weights is None: self.weights.append(random.uniform(min_val, max_val))
            else: self.weights.append(weights[i]*random.uniform(min_val, max_val))
        if bias is None: self.bias = random.uniform(min_val, max_val)
        else: self.bias = bias*random.uniform(min_val, max_val)

    def calc(self, inputs):
        self.output = 0
        for i in range(len(inputs)):
            self.output += inputs[i] * self.weights[i]
        self.output += self.bias
        self.output = 1 / (1 + math.exp(-self.output))
        return self.output
