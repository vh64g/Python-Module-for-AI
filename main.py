import random
import pickle

from ai import neuralNetwork
from ai import neuron as neuron
from ai.training import randomValueTraining
from ai.training import basicGradientDescent as basicGradientDescentTraining
from matplotlib import pyplot as plt


def analyze(rvt):
    for layer in rvt.best_neural_network.hidden_layers:
        print(f"Hidden layer: {layer}")
        for NEURON in layer:
            print(f"{NEURON}\n  Weights: {NEURON.weights}\n  Bias: {NEURON.bias}\n  Output: {NEURON.output}")


def create_data(length=100):
    training_data = []
    for _ in range(length):
        training_data.append(
            [
                [x := random.random(), y := random.random()],
                [1 if ((x*y) / 2) > 0.1 else 0]]
        )
    return training_data


def main():
    training_data = create_data(2000)
    test_data = create_data(10000)

    ann = neuralNetwork.artificialNeuralNetwork(
        input_layer=[0, 0],
        output_layer=[neuron.neuron() for _ in range(1)],
        hidden_layers=[
            [neuron.neuron() for _ in range(3)]
        ]
    )

    rvt = randomValueTraining.rVT01(ann, training_data, epochs=10000)
    ann = rvt.best_neural_network
    bgd = basicGradientDescentTraining.basicGradientDescentTraining(ann, training_data, test_data, epochs=1000, learning_rate=0.15, debug=True)
    ann = bgd.network

    fig, ax = plt.subplots()
    for _ in test_data:
        print(f"{round((test_data.index(_) / len(test_data)) * 100,2)}%")
        z = ann.calc(_[0])
        if z[0] < 0.5: ax.scatter(_[0][0], _[0][1], c="g")
        else: ax.scatter(_[0][0], _[0][1], c="y")
    plt.show()

    pickle.dump(ann, open("network.txt", "wb"))

if __name__ == '__main__':
    main()
