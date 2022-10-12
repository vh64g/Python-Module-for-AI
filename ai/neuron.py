import random

def sigmoid(output):
    return 1 / (1 + (2.7182818284590452353602874713527 ** -output))


class neuron:
    def __init__(self):
        self.weights = None
        self.costGradientWeights = []
        self.costGradientBias = None
        self.bias = 1
        self.output = None

    def randomize(self, input_count):
        new_weights = []
        for i in range(input_count): new_weights.append(random.uniform(-10, 10))
        self.bias = random.uniform(-10, 10)
        self.weights = new_weights

    def calc(self, inputs):
        self.output = 0
        for i in range(len(inputs)): self.output += inputs[i] * self.weights[i]
        self.output += self.bias
        self.output = sigmoid(self.output)
        return self.output

