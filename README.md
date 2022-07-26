# NN-for-digit-classification
In this project, I have released a basic neural network with a variable architecture. The neural network consists of m layers: 1 input layer, m - 2 hidden layers and 1 output layer. All layers will be fully connected.
The number of layers and their sizes are set when the object of the NNetwork class is initialized.
This neural network was created to classify the numbers 0 to 9 using a dataset called MNIST, which consists of 70,000 28 x 28 pixel images.
The dataset contains one label for each image indicating the number we see in each image.
To train the neural network, we will use stochastic gradient descent.