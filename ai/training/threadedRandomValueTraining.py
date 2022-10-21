import copy
import time
import threading


def loss(result, expected):
    """Calculate the loss"""
    loss01 = []
    for i in range(len(result)):
        loss01.append((result[i] - expected[i]) ** 2)
    return loss01


def train_data(data, neural_network):
    """Train the neural network with one data set"""

    result = neural_network.calc(data[0])  # calculate the output of the neural network with the input data
    expected = data[1]  # get the expected output

    return round(sum(loss(result, expected)), 10)  # return the loss


class rVT01:
    def __init__(self, neural_network, training_data, epochs=20000):

        self.network = neural_network

        self.training_epochs = []

        self.training_data = training_data
        self.epochs = epochs

        self.train()

    def train(self):
        for _ in range(self.epochs):
            self.randomize()
            network = copy.deepcopy(self.network)
            threading.Thread(target=self.train_epoch, args=[_, network]).start()
        self.set_best_neural_network()


    def train_epoch(self, epoch, neural_network):
        loss_sum_all = 0

        for data in self.training_data:
            loss_sum_all += train_data(data, neural_network)

        loss_sum_all = round(loss_sum_all, 10)
        self.training_epochs.append([epoch, loss_sum_all, neural_network])

    def randomize(self):
        """Randomize the weights of the neural network"""
        self.network.randomize()

    def set_best_neural_network(self):
        # Queue for the threads
        while len(self.training_epochs) < self.epochs:
            time.sleep(0.1)
        # Sort the training epochs by the loss
        self.training_epochs.sort(key=lambda x: x[1])
        self.network = copy.deepcopy(self.training_epochs[0][2])
        # Print the best neural network
        print(f"Best epoch: {self.training_epochs[0][0]}, Loss: {self.training_epochs[0][1]}")