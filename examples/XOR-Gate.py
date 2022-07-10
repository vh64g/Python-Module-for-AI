from ai import neuralNetwork
from ai import neuron
from ai.training import randomValueTraining


def main():
    training_data = [
        [[0, 0], [0]],
        [[0, 1], [1]],
        [[1, 0], [1]],
        [[1, 1], [0]]
    ]

    ann = neuralNetwork.artificialNeuralNetwork(
        input_layer=[0, 0],
        output_layer=[neuron.neuron() for i in range(1)],
        hidden_layers=[
            [neuron.neuron() for i in range(5)],
            [neuron.neuron() for i in range(10)],
            [neuron.neuron() for i in range(5)]
        ]
    )
    rvt = randomValueTraining.rVT01(ann, training_data, epochs=80000)

    while True:
        input_data = [
            float(input("Input x: ")),
            float(input("Input y: "))
        ]
        print(f"Output: {rvt.neural_network.calc(input_data)}")


if __name__ == '__main__':
    main()
