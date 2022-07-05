class rVT01:
    def __init__(self, neural_network, training_data, epochs=20000, learning_rate=0.1):

        self.neural_network = neural_network

        self.training_data = training_data
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.train()

    def train(self):
        for i in range(self.epochs):
            self.train_epoch()

    def train_epoch(self):
        for data in self.training_data:
            self.train_data(data)

    def train_data(self, data):
        for layer in self.neural_network.hidden_layers:
            for neuron in self.neural_network.hidden_layers[layer]:
                neuron.randomize()
