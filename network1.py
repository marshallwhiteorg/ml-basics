# derived from https://github.com/mnielsen/neural-networks-and-deep-learning

import numpy as np
import numpy.typing as npt

WILL_IMPLEMENT_TOMORROW = '''print("Will implement tomorrow.")'''


def sigmoid(z: npt.ArrayLike):
    ''' The sigmoid function. '''
    z = np.asarray(z)
    return 1/(1+np.exp(-z))

def sigmoid_prime(z: npt.ArrayLike):
    ''' Derivative of the sigmoid function. '''
    return sigmoid(z)*(1-sigmoid(z))


class Network1:
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

    
    def SGD(self, training_data: [], n_epochs: int,
        mini_batch_size: int, lr: float,
        validation_data: [] = None):
        ''' Performs stochastic gradient descent to train the network.

        alpha: learning rate
        training_data: list of (x, y) pairs
        validation_data: list of (x, y) pairs

        The y values in training_data and validation_data do not have the same
        shape. See mnist_loader.load_data_wrapper for details.

        If validation_data is not None, then we evaluate the network on this
        data after each epoch. This can slow down the algorithm a lot.
        '''
        n = len(training_data)
        if validation_data:
            n_validation = len(validation_data)

        for e_idx in range(n_epochs):
            self._rng.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size]
                            for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                #self.update_params(mini_batch, lr)
                self.update_mini_batch(mini_batch, lr)
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
        pass


    def my_evaluate(self, data: np.ndarray) -> int:
        ''' Returns the number of correct predictions by the network.
        
        Performs a forward pass through the network for each example
        in the data. We assume that the network's prediction is the index
        of the highest activation from the output layer.

        data: array of (x, y) pairs
        '''
        pass
        

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)