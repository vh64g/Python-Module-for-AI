class rVT01:
    def __init__(self, neural_network, training_data, epochs=20000, learning_rate=.1):

        self.neural_network = neural_network

        self.training_data = training_data
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.train()

    def train(self):
        for i in range(self.epochs):
            self.train_epoch(i)

    def train_epoch(self, epoch):
        for data in self.training_data:
            self.train_data(data, epoch)

    def train_data(self, data, epoch):
        result = self.neural_network.calc(data[0])
        expected = data[1]
        print(f"Training: Epoch: {epoch}, Result: {result}, Expected: {expected}")
        if result != expected:
            for layer in self.neural_network.hidden_layers:
                for neuron in self.neural_network.hidden_layers[0]:
                    neuron.randomize(len(self.neural_network.input_layer), -1*self.learning_rate, 1*self.learning_rate, neuron.weights, neuron.bias)
