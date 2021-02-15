import numpy as np

def percept(inputs: np.ndarray, weights: np.ndarray, threshold: float) -> int:
    ''' A perceptron.
    inputs: binary array
    weights: float array

    >>> percept(np.array([0, 1, 1]), np.array([1, 2, 3]), 4.0)
    1
    >>> percept(np.array([0, 1, 1]), np.array([1, 2, 3]), 5)
    0
    '''
    result = np.dot(inputs, weights)
    return int(result > threshold)


def percept_with_bias(inputs: np.ndarray, weights: np.ndarray, bias: float) -> int:
    ''' Like percept, but we use the concept of bias.
    Bias is -threshold.
    inputs: binary array
    weights: float array

    >>> percept_with_bias(np.array([0, 1, 1]), np.array([1, 2, 3]), -4)
    1
    >>> percept_with_bias(np.array([0, 1, 1]), np.array([1, 2, 3]), -5)
    0
    '''
    result = np.dot(inputs, weights) + bias
    return result > 0

def nand(a: bool, b: bool) -> bool:
    ''' NAND of a and b.
    >>> nand(False, False)
    True
    >>> nand(False, True)
    True
    >>> nand(True, False)
    True
    >>> nand(True, True)
    False
    '''
    result = percept_with_bias(np.array([int(a), int(b)]), np.array([-1, -1]), 2)
    return bool(result)


def sum_bits(x1: bool, x2: bool) -> bool:
    ''' Adds two bits. Returns bitwise sum and the carry bit.
    >>> sum_bits(False, False)
    (False, False)
    >>> sum_bits(False, True)
    (True, False)
    >>> sum_bits(True, False)
    (True, False)
    >>> sum_bits(True, True)
    (False, True)
    '''
    return (bitwise_sum(x1, x2), bitwise_and(x1, x2))


def bitwise_sum(x1: bool, x2: bool) -> bool:
    ''' Also known as x1 XOR x2
    >>> bitwise_sum(False, False)
    False
    >>> bitwise_sum(False, True)
    True
    >>> bitwise_sum(True, False)
    True
    >>> bitwise_sum(True, True)
    False
    '''
    b1 = nand(x1, x2)
    b2 = nand(x1, b1)
    b3 = nand(x2, b1)
    return nand(b2, b3)


def bitwise_and(x1: bool, x2: bool) -> bool:
    ''' Also known as x1 * x2.
    >>> bitwise_and(False, False)
    False
    >>> bitwise_and(False, True)
    False
    >>> bitwise_and(False, True)
    False
    >>> bitwise_and(True, True)
    True
    '''
    nand_1_2 = nand(x1, x2)
    return nand(nand_1_2, nand_1_2)


def sum_bits_faster(x1: bool, x2: bool) -> bool:
    ''' Faster version of sum_bits.
    >>> sum_bits_faster(False, False)
    (False, False)
    >>> sum_bits_faster(False, True)
    (True, False)
    >>> sum_bits_faster(True, False)
    (True, False)
    >>> sum_bits_faster(True, True)
    (False, True)
    '''
    b1 = nand(x1, x2)
    b2 = nand(x1, b1)
    b3 = nand(x2, b1)
    b_sum = nand(b2, b3)
    b_carry = nand(b1, b1)
    return (b_sum, b_carry)


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)