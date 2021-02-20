# derived from https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/mnist_svm.py

import mnist_loader
from sklearn import svm

def train_svm(X_train, Y_train, n_train):
    ''' Train a support vector machine using the given data. '''
    clf = svm.SVC()
    clf.fit(X_train[:n_train], Y_train[:n_train])
    return clf

def svm_baseline():
    n_train = 50000
    n_test = 10000
    (X_train, Y_train), (_, _), (X_test, Y_test) = mnist_loader.load_data()
    clf = train_svm(X_train, Y_train, n_train)
    predictions = [int(a) for a in clf.predict(X_test[:n_test])]
    num_correct = sum(int(a == y) for a, y in \
    zip(predictions, Y_test[:n_test]))
    print("Baseline classifier using an SVM.")
    print("%s of %s values correct." % (num_correct, len(Y_test[:n_test])))

if __name__ == "__main__":
    svm_baseline()