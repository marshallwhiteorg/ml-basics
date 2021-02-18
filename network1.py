# derived from https://github.com/mnielsen/neural-networks-and-deep-learning

import numpy as np

class Network:
    num_layers: int
    sizes: [int]
    biases: np.ndarray
    weights: np.ndarray
    def __init__(self, sizes: [int]):
        ''' sizes: size of each layer '''
        self.num_layers = len(sizes)
        self.sizes = sizes
        rng = np.random.default_rng()
        self.biases = [rng.standard_normal(size=(y, 1)) for y in self.sizes[1:]]
        # Each element is a matrix of weights where e.g. weights[0] is the
        # matrix of weights between layers 1 and 2. If layer 1 is size 3 and
        # layer 2 is size 4, then weights[0] is a 4x3 matrix where entry i,j
        # is the weight between neuron j in layer 1 and neuron i in layer 2.
        # The matrix has this shape so that we can easily multiply the weights
        # by the activations coming from the previous layer; the number of
        # columns must match the number of activations.
        self.weights = [rng.standard_normal(size=(second, first))
                        for first, second in zip(sizes[:-1], sizes[1:])]