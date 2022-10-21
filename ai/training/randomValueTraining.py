import copy
import time


def loss(result, expected):
    """Calculate the loss"""
    loss01 = []
    for i in range(len(result)):
        loss01.append((result[i] - expected[i]) ** 2)
    return loss01


class rVT01:
    def __init__(self, neural_network, training_data, epochs=20000):

        self.neural_network = neural_network
        self.best_neural_network = None
        self.training_epoch = 0

        self.lowest_loss = None

        self.training_data = training_data
        self.epochs = epochs

        self.train()

    def train(self):
        for i in range(self.epochs):
            self.randomize()
            self.train_epoch(i)
        self.set_best_neural_network()

    def train_epoch(self, epoch):
        loss_sum_all = 0

        for data in self.training_data:
            loss_sum_all += self.train_data(data, epoch)

        loss_sum_all = round(loss_sum_all, 10)

        if self.lowest_loss is None:  # if the loss is lower than the lowest loss
            print(f"Lowest loss caused by null loss: {loss_sum_all}")
            self.lowest_loss = loss_sum_all  # set the lowest loss to the current loss
            self.best_neural_network = copy.deepcopy(self.neural_network) # set the best neural network to the current neural network
            self.training_epoch = epoch  # set the training epoch to the current epoch
        elif loss_sum_all < self.lowest_loss:  # if the loss is lower than the lowest loss
            print(f"Lowest loss: {loss_sum_all} is lower then {self.lowest_loss}")
            self.lowest_loss = loss_sum_all  # set the lowest loss to the current loss
            self.best_neural_network = copy.deepcopy(self.neural_network)  # set the best neural network to the current neural network
            self.training_epoch = epoch  # set the training epoch to the current epoch

        print(f"Training: Epoch: {epoch}, Loss: {loss_sum_all}")

    def train_data(self, data):
        """Train the neural network with one data set"""

        result = self.neural_network.calc(data[0])  # calculate the output of the neural network with the input data
        expected = data[1]  # get the expected output

        return round(sum(loss(result, expected)), 10)  # return the loss

    def randomize(self):
        """Randomize the weights of the neural network"""
        self.neural_network.randomize()

    def set_best_neural_network(self):
        """Set the current neural network to the best neural network"""
        print("----------Setting best neural network----------")
        self.neural_network = copy.deepcopy(self.best_neural_network)
        time.sleep(1)

        print("----------Best neural network----------")
        # Calculate the loss of the best neural network

        lo = 0
        for data in self.training_data:
            result = self.neural_network.calc(data[0])  # calculate the output of the neural network with the input data
            expected = data[1]  # get the expected output
            lo += round(sum(loss(result, expected)), 10)
        print(f"Final network: Loss: {round(lo, 10)}, Epoch: {self.training_epoch}")


