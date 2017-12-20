"""Performs linear regression using gradient descent.

Linear regression can be solved directly, but here
we show how to calculate gradients and use simple
gradient descent."""

import numpy as np

NUM_EPOCHS = 100000

LEARNING_RATE = 0.005
LOSS_DISPLAY_FREQUENCY = 500

def gradw(x, yp, y): # weight gradient
    return x.T.dot(yp - y) / y.shape[0]

def gradb(yp, y): # bias gradient
    return (yp - y).sum() / y.shape[0]

def loss(yp, y): # loss (cost)
    l = (yp - y) * (yp - y)
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
        yp = forward(x, w, b)

        # calculate gradients
        gw = gradw(x, yp, y)
        gb = gradb(yp, y)
        
        # update weights
        w = w - LEARNING_RATE * gw
        b = b - LEARNING_RATE * gb

        # calculate loss in regular intervals
        if i % LOSS_DISPLAY_FREQUENCY == 0:
            l = loss(yp, y)
            print("Loss in {0}th epoch: {1:5f}".format(i, l))

    print("Final weights: w = {0}, b = {1}".format(w, b))


if __name__ == '__main__':
    main()