"""Performs linear regression using gradient descent.

Linear regression can be solved directly, but here
we show how to calculate gradients and use simple
gradient descent."""

import numpy as np

NUM_EPOCHS = 100000

LEARNING_RATE = 0.005
LOSS_DISPLAY_FREQUENCY = 500

def gradw(x, y_hat, y): # weight gradient
    return x.T.dot(y_hat - y) / y.shape[0]

def gradb(y_hat, y): # bias gradient
    return (y_hat - y).sum() / y.shape[0]

def loss(y_hat, y): # L2 loss (cost)
    l = (y_hat - y) * (y_hat - y)
    return -l.sum()

def forward(x, w, b): # forward pass through network
    return x.dot(w) + b

def main():
    # generate test data with x[:,0] strongly correlated with y and x[:,1] uncorrelateds
    # we expect w[0] = 0.33 and w[1] = 0, because x[:,0] already explains variability in y
    x = np.zeros((10, 2))
    x[:,0] = np.arange(3,31,3)
    x[:,1] = [5,9,6,1,10,3,2,4,7,8]
    y = np.arange(1,11)

    w = np.random.rand(2) * 100 # randomly initialize weights
    b = np.random.rand() * 100 # randomly initialize bias
    # note that we usually initialize b = 0, but here we want to check for correct functionality; b should converge to 0

    # perform gradient descent
    for i in range(1, NUM_EPOCHS):
        y_hat = forward(x, w, b)

        # calculate gradients
        gw = gradw(x, y_hat, y)
        gb = gradb(y_hat, y)
        
        # update weights
        w = w - LEARNING_RATE * gw
        b = b - LEARNING_RATE * gb

        # calculate loss in regular intervals
        if i % LOSS_DISPLAY_FREQUENCY == 0:
            l = loss(y_hat, y)
            print("Loss in {0}th epoch: {1:5f}".format(i, l))

    print("Final weights: w = {0}, b = {1}".format(w, b))

    # predict some numbers
    x_pred = np.array([[1,6],[2,7],[3,8],[4,9],[5,10]])
    y_pred = forward(x_pred, w, b)
    print("Predictions: {0}".format(y_pred))

if __name__ == '__main__':
    main()