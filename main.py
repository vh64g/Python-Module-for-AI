from ai import neuralNetwork
from ai import neuron
from ai.training import randomValueTraining


def main():
    ann = neuralNetwork.artificialNeuralNetwork(input_layer=[0, 0], output_layer=[neuron.neuron() for i in range(1)], hidden_layers=[[neuron.neuron() for i in range(5)]])
    rvt = randomValueTraining.rVT01(ann, [[[1, 0], [1]]], epochs=20000, learning_rate=0.1)


if __name__ == '__main__':
    main()