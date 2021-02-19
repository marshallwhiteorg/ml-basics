# derived from https://github.com/mnielsen/neural-networks-and-deep-learning

import numpy as np
import numpy.typing as npt

WILL_IMPLEMENT_TOMORROW = '''print("Will implement tomorrow.")'''


def sigmoid(z: npt.ArrayLike):
    return 1/(1 + np.exp(-1 * np.asarray(z)))


class Network:
    num_layers: int
    sizes: [int]
    biases: np.ndarray
    weights: np.ndarray
    _rng = np.random.default_rng()

    def __init__(self, sizes: [int]):
        ''' sizes: size of each layer '''
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [self._rng.standard_normal(size=(y, 1)) for y in self.sizes[1:]]
        # Each element is a matrix of weights where e.g. weights[0] is the
        # matrix of weights between layers 1 and 2. If layer 1 is size 3 and
        # layer 2 is size 4, then weights[0] is a 4x3 matrix where entry i,j
        # is the weight between neuron j in layer 1 and neuron i in layer 2.
        # The matrix has this shape so that we can easily multiply the weights
        # by the activations coming from the previous layer; the number of
        # columns must match the number of activations.
        self.weights = [self._rng.standard_normal(size=(second, first))
                        for first, second in zip(sizes[:-1], sizes[1:])]


    # TODO consider MN's comment about modifying to feedforward multiple inputs
    # at once using (n, 1) ndarrays rather than (n,) vectors.
    def feedforward(self, inputs: npt.ArrayLike) -> np.ndarray:
        ''' Returns the output of the network on the given inputs. '''
        a = np.asarray(inputs)
        assert a.size == self.sizes[0]

        for w, b in zip(self.weights, self.biases):
            a = sigmoid(w@a + b)
        return a

    
    def SGD(self, training_data: npt.ArrayLike, n_epochs: int, lr: int,
            mini_batch_size: int, validation_data: npt.ArrayLike = None):
        ''' Performs stochastic gradient descent to train the network.

        alpha: learning rate
        training_data: array of (x, y) pairs of training data

        If validation_data is not None, then we evaluate the network on this
        data after each epoch. This can slow down the algorithm a lot.
        '''
        training_data = np.asarray(training_data)
        if validation_data:
            validation_data = np.asarray(validation_data)
            n_validation = len(validation_data)
        n = len(training_data)

        for e_idx in range(n_epochs):
            self._rng.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size]
                            for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_params(mini_batch, lr)
            if validation_data:
                n_correct = self.evaluate(validation_data)
                print("Epoch {0}: {1} / {2}".format(
                    e_idx, n_correct, n_validation))
            else:
                print("Epoch {0} complete.".format(e_idx))

    def update_params(self, mini_batch: np.ndarray, lr):
        ''' Updates weights and biases using gradient descent on a single
        mini-batch. The gradient of the cost function is calculated
        via backpropagation. 

        mini_batch: array of (x, y) pairs of training data
        lr: learning rate
        '''
        exec(WILL_IMPLEMENT_TOMORROW)


    def evaluate(self, data: np.ndarray) -> int:
        ''' Returns the number of correct predictions by the network.
        
        Performs a forward pass through the network for each example
        in the data. We assume that the network's prediction is the index
        of the highest activation from the output layer.

        data: array of (x, y) pairs
        '''
        exec(WILL_IMPLEMENT_TOMORROW)