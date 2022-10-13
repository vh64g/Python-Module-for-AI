from ai import neuralNetwork
from ai import neuron as neuron
from ai.training import randomValueTraining
from ai.training import basicGradientDescent as basicGradientDescentTraining


def analyze(rvt):
    for layer in rvt.best_neural_network.hidden_layers:
        print(f"Hidden layer: {layer}")
        for NEURON in layer:
            print(f"{NEURON}\n  Weights: {NEURON.weights}\n  Bias: {NEURON.bias}\n  Output: {NEURON.output}")


def main():
    training_data = [
        [[0, 0], [0]],
        [[0, 1], [1]],
        [[1, 0], [1]],
        [[1, 1], [0]]
    ]

    ann = neuralNetwork.artificialNeuralNetwork(
        input_layer=[0, 0],
        output_layer=[neuron.neuron() for _ in range(1)],
        hidden_layers=[
            [neuron.neuron() for _ in range(3)]
        ]
    )
    bgd = basicGradientDescentTraining.basicGradientDescentTraining(ann, training_data, epochs=1000, learning_rate=0.2)
    ann = bgd.network
    res = ann.calc([[1, 0],[1]])
    print(res)
    # rvt = randomValueTraining.rVT01(ann, training_data, epochs=10000)


if __name__ == '__main__':
    main()
