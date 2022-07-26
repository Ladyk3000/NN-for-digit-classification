import numpy as np
from load_mnist import load_data
import time


class NNetwork:

    def __init__(self, sizes, epochs=10, l_rate=0.001):
        self.sizes = sizes
        self.epochs = epochs
        self.l_rate = l_rate
        self.layers = len(sizes)
        print(f'count of layers :{self.layers}')
        self.params = self.initialization()
        #print(self.params)
    
    def sigmoid(self, x, derivative=False):
        if derivative:
            return (np.exp(-x))/((np.exp(-x)+1)**2)
        return 1/(1 + np.exp(-x))
    
    def softmax(self, x, derivative=False):
        exps = np.exp(x - x.max())
        if derivative:
            return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
        return exps / np.sum(exps, axis=0)
    
    def initialization(self):
        layers = self.layers

        layers_size = []
        layers_size.append(self.sizes[0])
        for i in range(1,self.layers - 1):
            layers_size.append(self.sizes[i])
        layers_size.append(self.sizes[-1])

        params = {}
        for j in range(1, layers):
            params[f'W{j}'] = np.random.randn(layers_size[j], layers_size[j - 1]) * np.sqrt(1. / layers_size[j])
        
        return params

    def forward_pass(self, x_train):
        params = self.params
        layers = self.layers
        params['A0'] = x_train

        for j in range(1, layers):
            params[f'Y{j}'] = np.dot(params[f'W{j}'], params[f'A{j - 1}'])
            params[f'A{j}'] = self.sigmoid(params[f'Y{j}'])
        
        return params[f'A{layers - 1}']

    def backward_pass(self, y_train, output):
        layers = self.layers - 1
        params = self.params
        weights_changes = {}
        
        error = 2 * (output - y_train) / output.shape[0] * self.softmax(params[f'Y{layers}'], derivative=True)
        weights_changes[f'W{layers}'] = np.outer(error, params[f'A{layers - 1}'])
        
        layers = sorted(range(1,layers), reverse=True)
        for j in layers:
            error = np.dot(params[f'W{j + 1}'].T, error) * self.sigmoid(params[f'Y{j}'], derivative=True)
            weights_changes[f'W{j}'] = np.outer(error, params[f'A{j - 1}'])
            
        return weights_changes

    def update_network_parameters(self, changes_to_w):
        for key, value in changes_to_w.items():
            self.params[key] -= self.l_rate * value

    def compute_accuracy(self, x_val, y_val):
        predictions = []

        for x, y in zip(x_val, y_val):
            output = self.forward_pass(x)
            pred = np.argmax(output)
            predictions.append(pred == np.argmax(y))
        
        return np.mean(predictions)

    def train(self, x_train, y_train, x_val, y_val):
        start_time = time.time()
        for iteration in range(self.epochs):
            for x,y in zip(x_train, y_train):
                output = self.forward_pass(x)
                changes_to_w = self.backward_pass(y, output)
                self.update_network_parameters(changes_to_w)
            
            accuracy = self.compute_accuracy(x_val, y_val)
            print('Epoch: {0}, Time Spent: {1:.2f}s, Accuracy: {2:.2f}%'.format(
                iteration+1, time.time() - start_time, accuracy * 100
            ))

def main():
    X_train, y_train = load_data(r'Neural Network Base\data\mnist_train.csv')
    X_test, y_test = load_data(r'Neural Network Base\data\mnist_test.csv')
    dnn = NNetwork(sizes=[784, 128, 64, 10])
    dnn.train(X_train, y_train, X_test, y_test)

if __name__ == '__main__':
    main()