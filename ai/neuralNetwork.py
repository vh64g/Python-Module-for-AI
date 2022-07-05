class artificialNeuralNetwork:
    def __init__(self, input_layer, output_layer, hidden_layers=None):

        self.input_layer = input_layer  # structure of input layer: [neuron1, neuron2, ...]
        self.hidden_layers = hidden_layers  # structure of hidden layers: [[layer1_neurons], [layer2: neurons, ...]
        self.output_layer = output_layer  # structure of output layer: [neuron1, neuron2, ...]

        if hidden_layers is None: self.hidden_layers = {}

        self.out = []

        self.randomize()

    def randomize(self):
        for layer in self.hidden_layers:
            for neuron in self.hidden_layers[0]:
                neuron.randomize(len(self.input_layer))
        for neuron in self.output_layer:
            neuron.randomize(len(self.hidden_layers[-1]))

    def calc(self, inputs):
        self.input_layer = inputs
        self.out = []
        for layer in self.hidden_layers:
            for neuron in self.hidden_layers[0]:
                x = neuron.calc(self.input_layer)
        for neuron in self.output_layer:
            middle_outs = [neuron.output for neuron in self.hidden_layers[-1]]
            self.out.append(neuron.calc(middle_outs))
        return self.out
