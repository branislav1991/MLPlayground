"""Performs logistic regression using gradient descent."""

import numpy as np

NUM_EPOCHS = 1000000

LEARNING_RATE = 0.01
LOSS_DISPLAY_FREQUENCY = 10000

def logistic(x): # logistic function
    return 1.0 / (1.0 + np.exp(-x))

def gradxentropy(y_hat, y):
    return ((1-y)/(1-y_hat))-(y/y_hat)

def gradlogistic(x): # gradient of the logistic function
    return logistic(x) * (1.0 - logistic(x))

def gradw(x, z, y_hat, y): # weight gradient
    outer_grads = gradxentropy(y_hat,y) * gradlogistic(z)
    return x.T.dot(outer_grads)

def gradb(z, y_hat, y): # bias gradient
    grads = gradxentropy(y_hat,y) * gradlogistic(z)
    return grads.mean(axis=0)

def loss(y_hat, y): # cross entropy loss (cost)
    y_hat = np.minimum(np.maximum(y_hat, 1e-5), 1.0 - 1e-5) # avoid division by zero
    l = -y * np.log(y_hat) - (1-y) * np.log(1-y_hat)
    return l.sum()

def accuracy(y_hat, y):
     correct_prediction = np.equal(np.argmax(y_hat, axis=1), np.argmax(y, axis=1))
     return correct_prediction.mean()

def forward(x, w, b): # forward pass through network
    return x.dot(w) + b

def main():
    x = np.zeros((10, 2)) # 2-dimensional input; first dimension = batch size
    x[:,0] = np.random.rand(10)
    x[:,1] = [100.0, 50.0, 60.0, 30.0, 80.0, 6.0, 70.0, 20.0, 90.0, 70.0]
    x[:,1] = x[:,1] / 100.0

    y = np.zeros((10, 2))
    y[:,0] = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0] # 2-class output (one-hot encoding)
    y[:,1] = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0] # 2-class output (one-hot encoding)

    w = np.random.rand(2, 2) # randomly initialize weights
    b = np.zeros(2) # initialize biases to zero

    # perform gradient descent
    for i in range(1, NUM_EPOCHS):
        z = forward(x, w, b)
        y_hat = logistic(z) # prediction

        # calculate gradients
        gw = gradw(x, z, y_hat, y)
        gb = gradb(z, y_hat, y)
        
        # update weights
        w = w - LEARNING_RATE * gw
        b = b - LEARNING_RATE * gb

        # calculate loss in regular intervals
        if i % LOSS_DISPLAY_FREQUENCY == 0:
            l = loss(y_hat, y)
            a = accuracy(y_hat, y)
            print("Loss in {0}th epoch: {1:5f}, Accuracy: {2}".format(i, l, a))

    print("Final weights: w = {0}, b = {1}".format(w, b))

    # predict some numbers
    x_pred = np.array([[1,6],[2,7],[3,8],[4,9],[5,10]])
    x_pred = x_pred / 33.0
    y_pred = np.argmax(forward(x_pred, w, b), axis=1)
    print("Predictions: {0}".format(y_pred))

if __name__ == '__main__':
    main()