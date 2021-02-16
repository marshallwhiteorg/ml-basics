import numpy as np
import math


def sigmoid_act(inputs, weights, bias: float) -> float:
    ''' Sigmoid activation function.
    inputs: array_like
    weights: array_like

    Let a = np.dot(inputs, weights) + b.
    If a is 0, sigmoid is .5.
    If a is large negative, sigmoid is close to 0.
    If a is large positive, sigmoid is close to 1.

    >>> sigmoid_act([2, 3], [1, 1], -5)
    0.5
    '''
    z = np.dot(np.array(inputs), np.array(weights)) + bias
    return 1/(1 + math.exp(-z))


def digit_activations(digit: int) -> [float]:
    ''' Simulates the activations of a 10 neuron output layer used to
    classify digits. The neuron with the correct digit has an
    output of at least .99, and the incorrect outputs have activation < .01.

    For simplicity, if the correct digit in 0..9 is i, we output an array A where A[i]
    is .995 and the other elements are 0.005.
    >>> digit_activations(0)
    [0.995, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005]
    >>> digit_activations(9)
    [0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.995]
    '''
    a = [0.005] * 10
    a[digit] = .995
    return a


def binary_activations(digit_acts) -> [float]:
    ''' Simulates the activations of the 4 neuron binary layer
    given the output of the 10 neuron digit layer.
    
    Outputs an array A of size 4 where each element is the activation
    for that bit. A[0] is the activation for the most significant bit, and
    A[3] is the activation for the LSB.
    '''
    act_b1 = sigmoid_act(digit_acts, [0, 1, 0, 1, 0, 1, 0, 1, 0, 1], -.05) # LSB
    act_b2 = sigmoid_act(digit_acts, [0, 0, 1, 1, 0, 0, 1, 1, 0, 0], -.04)
    act_b3 = sigmoid_act(digit_acts, [0, 0, 0, 0, 1, 1, 1, 1, 0, 0], -.04)
    act_b4 = sigmoid_act(digit_acts, [0, 0, 0, 0, 0, 0, 0, 0, 1, 1], -.02) # MSB
    return [act_b4, act_b3, act_b2, act_b1]


def classify_output(bin_acts) -> [int]:
    ''' Given the activations of the binary layer,
    gives the predicted binary representation.

    Activations must be in [0, .5) or (.5, 1].
    '''
    return [int(x > .5) for x in bin_acts]


def predict_binary(digit) -> str:
    ''' Predicts the binary representation of the digit using
    the network of sigmoid neurons.
    >>> predict_binary(0)
    '0b0000'
    >>> predict_binary(1)
    '0b0001'
    >>> predict_binary(2)
    '0b0010'
    >>> predict_binary(3)
    '0b0011'
    >>> predict_binary(4)
    '0b0100'
    >>> predict_binary(5)
    '0b0101'
    >>> predict_binary(6)
    '0b0110'
    >>> predict_binary(7)
    '0b0111'
    >>> predict_binary(8)
    '0b1000'
    >>> predict_binary(9)
    '0b1001'
    '''
    bits = classify_output(binary_activations(digit_activations(digit)))
    n = 0
    for (idx, bit) in enumerate(bits):
        n += bit * (2**(3-idx)) # use 3-idx because the first bit is msb, last is lsb
    return format(n, '#06b')


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)