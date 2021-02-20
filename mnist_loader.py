# derived from https://github.com/mnielsen/neural-networks-and-deep-learning

import pickle
import gzip
import numpy as np

def load_data() -> (np.ndarray, np.ndarray, np.ndarray):
    ''' Returns the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    training_data is a tuple with two entries.
    The first entry is an ndarray of length 50,000 where each entry is an
    ndarray with 784 values, representing the pixels in one MNIST image.
    The second entry is an ndarray containing the 50,000 labels, where each
    label is a digit 0...9.

    validation_data and test_data have the same structure, except the lengths
    are 10,000.
    '''
    f = gzip.open('./data/mnist.pkl.gz', 'rb')
    # There is some issue with the encoding of the file, but latin1 seems
    # to work.
    training_data, validation_data, test_data = pickle.load(
        f, encoding='latin1')
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper() -> (np.ndarray, np.ndarray, np.ndarray):
    ''' Returns a tuple of MNIST data as (training_data, validation_data,
    test_data). Like load_data, but the format is altered for easier use.

    In particular, training_data is a list containing 50,000
    (x, y) pairs. x is a 784-dimensional ndarray
    containing the input image. y is a 10-dimensional
    ndarray representing the unit vector corresponding to the
    correct digit for x.

    validation_data and test_data are lists containing 10,000
    (x, y) pairs.  In each case, x is a 784-dimensional
    ndarray containing the input image, and y is the
    corresponding classification, i.e., the digit
    corresponding to x.
    
    Note the difference in format between the lists. training_data
    encodes the digit as a 10d vector with a 1.0 in the digit's position,
    and validation_data and test_data encode the digit as an integer.

    Additionally, in all three cases x has shape (784, 1), not (784,),
    as it is in load_data.
    Apparently this is convenient.
    '''
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [_vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    return (training_data, validation_data, test_data)

def _vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

