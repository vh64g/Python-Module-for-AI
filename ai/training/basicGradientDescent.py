from matplotlib import pyplot as plt


class basicGradientDescentTraining:
    def __init__(self, ann, training_data, test_data=None, epochs=1000, learning_rate=0.2, debug=False):
        if test_data is None:
            self.test_data = training_data
        self.network = ann
        self.training_data = training_data
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.debug = debug
        self.epochsList = []
        self.costs = []
        self.run()

    def debug_func(self):
        # create error graph
        fig, ax = plt.subplots()
        ax.set_title(f"Cost of network over {self.epochs} epochs")
        ax.grid(True)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cost")
        ax.set_xlim(0, self.epochs)
        ax.set_ylim(0, max(self.costs))
        ax.plot(self.epochsList, self.costs)

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
        self.costs = []
        self.epochsList = []
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
            cost = self.cost()
            self.costs.append(cost)
            self.epochsList.append(_)
            print(f"EPOCH: {_} | COST: {cost}")
        if self.debug:
            self.debug_func()

    def cost(self):
        cost = 0
        for data in self.training_data:
            cost += abs(sum(self.network.calc(data[0])) - abs(sum(data[1])))
        return abs(cost/len(self.training_data))
