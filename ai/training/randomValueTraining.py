import copy


class rVT01:
    def __init__(self, neural_network, training_data, epochs=20000):

        self.neural_network = neural_network
        self.best_neural_network = None

        self.lowest_loss = None

        self.training_data = training_data
        self.epochs = epochs

        self.train()

    def train(self):
        for i in range(self.epochs):
            self.train_epoch(i)
        self.set_best_neural_network()

    def train_epoch(self, epoch):
        loss_sum_all = 0

        for data in self.training_data:
            loss_sum_all += self.train_data(data, epoch)

        if self.lowest_loss is None:  # if the loss is lower than the lowest loss
            print(f"Lowest loss: {loss_sum_all}")
            self.lowest_loss = loss_sum_all  # set the lowest loss to the current loss
            self.best_neural_network = copy.deepcopy(self.neural_network)  # set the best neural network to the current neural network
        elif self.lowest_loss > loss_sum_all:  # if the loss is lower than the lowest loss
            print(f"Lowest loss: {loss_sum_all}")
            self.lowest_loss = loss_sum_all  # set the lowest loss to the current loss
            self.best_neural_network = copy.deepcopy(self.neural_network)  # set the best neural network to the current neural network

        print(f"Training: Epoch: {epoch}, Loss: {loss_sum_all}")

    def train_data(self, data, epoch):
        """Train the neural network with one data set"""
        self.neural_network.randomize()  # randomize the weights of the neural network

        result = self.neural_network.calc(data[0])  # calculate the output of the neural network with the input data
        expected = data[1]  # get the expected output
        loss = [abs(expected[i] - result[i]) for i in range(len(result))]  # calculate the loss
        loss_sum = sum(loss)  # sum the loss

        return loss_sum

    def set_best_neural_network(self):
        """Set the current neural network to the best neural network"""
        print("----------Setting best neural network----------")
        self.neural_network = copy.deepcopy(self.best_neural_network)
        """Calculate the output of the neural network with the input data"""
        for data in self.training_data:
            result = self.neural_network.calc(data[0])
            expected = data[1]
            loss = [abs(expected[i] - result[i]) for i in range(len(result))]
            print(f"Final network: Result: {result}, Expected: {expected}, Loss: {loss}")

