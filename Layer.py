import numpy
import numpy as np
import csv


class Layer:
    def __init__(self, n_count=None, prev_count=None, neurons=None, weights=None, bias=None, prev_layer=None, next_layer=None):
        if neurons is None:
            self.neurons: np.array = np.zeros(n_count)
        else:
            self.neurons = neurons
        # Gives us n x prev matrix for all the weights
        if weights is None:
            self.weights = np.random.uniform(-1, 2, (n_count, prev_count))
        else:
            self.weights = weights

        if bias is None:
            self.bias = np.zeros(n_count)
        else:
            self.bias = bias

        self.prev_layer: Layer = prev_layer

        self.next_layer: Layer = next_layer

    def add_layer(self, layer):
        # Find last layer
        current = self
        while current.next_layer:
            current = current.next_layer

        # Update last layer
        current.next_layer = layer
        layer.prev_layer = current

    def process(self, data=None):
        if data is None:
            neurons = self.prev_layer.neurons
            new_neurons = np.matmul(self.weights, neurons) + self.bias

            # Map all negative values to zero
            def map_to_bounds(i):
                if i <= 0:
                    return 0
                elif i >= 1:
                    return 1
                else:
                    return i

            self.neurons = np.vectorize(map_to_bounds)(new_neurons)

        else:
            self.neurons = np.array(tuple(data))

        if self.next_layer:
            return self.next_layer.process()
        else:
            return self.neurons

    def save(self, path):
        if not path.endswith('csv'):
            path = f'{path}.csv'
        with open(path, 'w') as file:
            current = self
            while current:
                current.neurons.tofile(file, sep=',')
                file.write('\n')
                current.weights.tofile(file, sep=',')
                file.write('\n')
                current.bias.tofile(file, sep=',')
                file.write('\n')

                current = current.next_layer

    @staticmethod
    def load(path):
        if not path.endswith('csv'):
            path = f'{path}.csv'
        layer = None
        neurons = []
        weights = []
        bias = []
        with open(path, 'r') as file:
            lines = file.readlines()
            property_amount = 3
            n_count = 0
            prev_n = 0
            for i in range(len(lines)):
                # Neurons
                if i % property_amount == 0:
                    neurons = np.fromstring(lines[i], sep=',', dtype=float)
                    prev_n = n_count
                    n_count = neurons.shape[0]

                # Weights
                elif i % property_amount == 1:
                    weights = []

                    # Extract the 2d array from a single line using current and previous neuron sizes
                    if not lines[i][0] == '\n':
                        for j in range(n_count):
                            arr = np.array(lines[i].replace('\n', '').split(',')[j * prev_n:(j + 1) * prev_n], dtype=float)
                            weights.append(arr)
                    if not weights:
                        weights = np.zeros(shape=(n_count, prev_n))
                    else:
                        weights = np.array(weights)

                # Biases
                elif i % property_amount == 2:
                    bias = np.fromstring(lines[i], sep=',', dtype=float)

                    new_layer = Layer(neurons=neurons, weights=weights, bias=bias)

                    if layer is None:
                        layer = new_layer
                    else:
                        layer.add_layer(new_layer)
        return layer

    def back_propagation(self):
        pass

    @staticmethod
    def load_dataset(path, res_start=1):
        with open(path) as file:
            csv_reader = csv.reader(file)

            def to_int(lst):
                for index in range(len(lst)):
                    lst[index] = int(lst[index])
                return lst

            results = to_int(next(csv_reader))
            data = []
            for row in csv_reader:
                data.append(to_int(row))

        dataset = []
        for i in range(len(results)):
            #     dataset[i] = dataset[i].split(',')
            #     results = []
            #     for j in range(res_start):
            #         results.append(int(dataset[i][j]))
            #
            #     inp = []
            #     for j in range(res_start, len(dataset[i])):
            #         inp.append(int(dataset[i][j]))

            dataset.append((results[i], data[i]))
        print(dataset[res_start - 1])

        return dataset

    def __str__(self):
        st = (f'Neurons:{self.neurons}\n'
              # f'Shape:{self.neurons.shape}\n'
              f'Weights:{self.weights}\n'
              # f'Shape:{self.weights.shape}\n'
              f'Next Layer:{self.next_layer}\n')
        return st

    def __eq__(self, other):
        if isinstance(other, Layer):
            n = numpy.array_equal(self.neurons, other.neurons)
            w = numpy.array_equal(self.weights, other.weights)
            b = numpy.array_equal(self.bias, other.bias)
            return n and w and b and self.next_layer == other.next_layer
