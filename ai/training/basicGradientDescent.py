class basicGradientDescentTraining:
    def __init__(self, ann, training_data, test_data=None, epochs=1000, learning_rate=0.2, debug=False):
        if test_data is None:
            self.test_data = training_data
        self.network = ann
        self.training_data = training_data
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.debug = debug
        if self.debug: from matplotlib import pyplot as plt
        self.run()

    def debug(self):
        # create error graph
        pass

    def train(self):
        h = 0.0001
        o_cost = self.cost()
        for hidden_layer in self.network.hidden_layers:
            for neuron in hidden_layer:
                self.calc_gradient(neuron, o_cost, h)
        for neuron in self.network.output_layer:
            self.calc_gradient(neuron, o_cost, h)

    def calc_gradient(self, neuron, o_cost, h):
        for weight in neuron.weights:
            weight += h
            n_cost = self.cost()
            neuron.costGradientWeights.append((n_cost - o_cost) / h)
            weight -= h
        neuron.bias += h
        n_cost = self.cost()
        neuron.costGradientBias = (n_cost - o_cost) / h
        neuron.bias -= h

    def run(self):
        for _ in range(self.epochs):
            self.train()
            for hidden_layer in self.network.hidden_layers:
                for neuron in hidden_layer:
                    for weight in neuron.weights:
                        neuron.weights[neuron.weights.index(weight)] -= self.learning_rate * neuron.costGradientWeights[neuron.weights.index(weight)]
                    neuron.bias -= self.learning_rate * neuron.costGradientBias
            for neuron in self.network.output_layer:
                for weight in neuron.weights:
                    neuron.weights[neuron.weights.index(weight)] -= self.learning_rate * neuron.costGradientWeights[neuron.weights.index(weight)]
                neuron.bias -= self.learning_rate * neuron.costGradientBias

            print(f"EPOCH: {_} | COST: {self.cost()}")

    def cost(self):
        cost = 0
        for data in self.training_data:
            cost += abs(sum(self.network.calc(data[0])) - abs(sum(data[1])))
        return abs(cost/len(self.training_data))
