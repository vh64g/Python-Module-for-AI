from ai import neuralNetwork
from ai import neuron
from ai.training import randomValueTraining


def main():
    training_data = [
        [[1], [1]],
        [[.5], [.5]],
        [[0], [0]]
    ]

    ann = neuralNetwork.artificialNeuralNetwork(
        input_layer=[0],
        output_layer=[neuron.neuron() for i in range(1)],
        hidden_layers=[
            [neuron.neuron() for i in range(1)]
        ]
    )
    rvt = randomValueTraining.rVT01(ann, training_data, epochs=2000000, learning_rate=0.1)

    while True:
        input_data = [
            float(input("Input x: "))
        ]
        print(f"Output: {rvt.neural_network.calc(input_data)}")


if __name__ == '__main__':
    main()
