import network1
import mnist_loader
import importlib

def test_network1():
    training_data, validation_data, test_data = \
    mnist_loader.load_data_wrapper()
    net = network1.Network1([784, 30, 10])
    net.SGD(training_data, 30, 10, 3.0, validation_data=validation_data)

def reload(module):
    importlib.reload(module)