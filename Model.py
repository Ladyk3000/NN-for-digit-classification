import numpy as np
from load_mnist import load_data
import time


class NNetwork:

    def __init__(self, sizes, epochs=3, learning_rate=0.01):
        self.sizes = sizes
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.layers = len(sizes)
        self.params = self.initialization()
    
    def print_architechture(self):
        print(f'Count of layers : {self.layers}')
        sizes = self.sizes
        print(f'Layers: {sizes[0]}', end = ' ')
        for i in range(1, len(sizes)):
            print(f'-> {sizes[i]}', end = ' ')
        print()

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def d_sigmoid(self, x):
        return (np.exp(-x))/((np.exp(-x)+1)**2)
    
    def softmax(self, x):
        exps = np.exp(x - x.max())
        return exps / np.sum(exps, axis=0)
    
    def d_softmax(self, x):
        exps = np.exp(x - x.max())
        return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))

    def compute_loss(self, weights,pred_error, output):
        loss = np.dot(weights.T, pred_error) * self.d_sigmoid(output)
        return loss
    
    def compute_chain_derivate(self, y_pred, y_train, Y_layer):
        derivate = 2 * (y_pred - y_train) / y_pred.shape[0] * self.d_softmax(Y_layer)
        return derivate

    def compute_weights_changes(self, A, B):
        changes = np.outer(A, B)
        return changes

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
        
        error = self.compute_chain_derivate(output, y_train)
        weights_changes[f'W{layers}'] = self.compute_weights_changes(error, params[f'A{layers - 1}'])
        
        layers = sorted(range(1,layers), reverse=True)
        for j in layers:
            error = self.compute_loss(params[f'W{j + 1}'], error, params[f'Y{j}'])
            weights_changes[f'W{j}'] = self.compute_weights_changes(error, params[f'A{j - 1}'])
            
        return weights_changes

    def update_parameters(self, changes_to_w):
        for key, value in changes_to_w.items():
            self.params[key] -= self.learning_rate * value

    def compute_accuracy(self, x_val, y_val):
        predictions = []

        for x, y in zip(x_val, y_val):
            output = self.forward_pass(x)
            pred = np.argmax(output)
            predictions.append(pred == np.argmax(y))
       
        accuracy = np.mean(predictions) * 100
        return accuracy

    def train(self, x_train, y_train, x_val, y_val):
        start_time = time.time()
        for epoch in range(self.epochs):
            for x,y in zip(x_train, y_train):
                output = self.forward_pass(x)
                changes_to_w = self.backward_pass(y, output)
                self.update_parameters(changes_to_w)
            
            accuracy = self.compute_accuracy(x_val, y_val)

            print(f'Epoch: {epoch+1}, Time Spent: {time.time() - start_time:.2f}sec, Accuracy: {accuracy:.2f}%')
    
def main():
    X_train, y_train = load_data(r'Neural Network Base\data\mnist_train.csv')
    X_test, y_test = load_data(r'Neural Network Base\data\mnist_test.csv')
    dnn = NNetwork(sizes=[784, 128, 10])
    dnn.print_architechture()
    dnn.train(X_train, y_train, X_test, y_test)

if __name__ == '__main__':
    main()