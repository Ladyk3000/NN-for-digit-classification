import numpy as np


def load_data(path):
    def one_hot_encoding(y):
        table = np.zeros((y.shape[0], 10))
        for i in range(y.shape[0]):
            table[i][int(y[i][0])] = 1 
        return table

    def normalize(x):
        x = (x/255).astype('float32')
        return x 

    data = np.loadtxt('{}'.format(path), delimiter = ',')
    return normalize(data[:,1:]),one_hot_encoding(data[:,:1])
