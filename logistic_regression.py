"""Performs logistic regression using stochastic gradient descent."""

import numpy as np

NUM_CLASSES = 2
NUM_FEATURES = 3

NUM_EPOCHS = 100000
NUM_EXAMPLES = 10

LEARNING_RATE = 0.001
LOSS_DISPLAY_FREQUENCY = 1000

def logistic(x): # logistic function
    return 1.0 / (1.0 + np.exp(-x))

def gradxentropy(y_hat, y):
    y_hat = np.minimum(np.maximum(y_hat, 1e-5), 1.0 - 1e-5) # avoid division by zero
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
    y_hat = np.minimum(np.maximum(y_hat, 1e-5), 1.0 - 1e-5) # avoid logarithm of zero
    l = -y * np.log(y_hat) - (1-y) * np.log(1-y_hat)
    return l.sum()

def accuracy(y_hat, y):
     correct_prediction = np.equal(np.argmax(y_hat, axis=1), np.argmax(y, axis=1)).astype(int)
     correct_prediction = correct_prediction.astype(float)
     return correct_prediction.mean()

def forward(x, w, b): # forward pass through network
    return x.dot(w) + b

def main():
    x = np.zeros((NUM_EXAMPLES, NUM_FEATURES)) # 2-dimensional input; first dimension = batch size
    x[:,0] = [-2.0, 6.0, -3.0, 2.0, -9.0, 2.0, -1.0, 10.0, -7.0, 7.0]
    x[:,1] = [10.0, -5.0, 6.0, -3.0, 8.0, -6.0, 7.0, -2.0, 9.0, -7.0]
    x[:,2] = [0.0, 0.0, 0.0, 1.0, -9.0, 6.0, -5.0, 2.0, -9.0, 1.0]
    x = x / 10.0

    y = np.zeros((NUM_EXAMPLES, NUM_CLASSES))
    y[:,0] = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0] # 2-class output (one-hot encoding)
    y[:,1] = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0] # 2-class output (one-hot encoding)

    w = np.random.rand(NUM_FEATURES, NUM_CLASSES) # randomly initialize weights
    b = np.zeros(NUM_CLASSES) # initialize biases to zero

    # perform stochastic gradient descent (no batch, just 1 example per iteration)
    for i in range(0, NUM_EPOCHS):
        for j in range(0, NUM_EXAMPLES):
            x_ex = np.reshape(x[j,:], (1,NUM_FEATURES))
            y_ex = np.reshape(y[j,:], (1,NUM_CLASSES))

            z = forward(x_ex, w, b)
            y_hat = logistic(z) # prediction

            # calculate gradients
            gw = gradw(x_ex, z, y_hat, y_ex)
            gb = gradb(z, y_hat, y_ex)
            
            # update weights
            w = w - LEARNING_RATE * gw
            b = b - LEARNING_RATE * gb

        # calculate loss in regular intervals
        if i % LOSS_DISPLAY_FREQUENCY == 0:
            y_hat = logistic(forward(x, w, b))
            l = loss(y_hat, y)
            a = accuracy(y_hat, y)
            print("Loss in {0}th epoch: {1:5f}, Accuracy: {2}".format(i, l, a))

    print("Final weights: w = {0}, b = {1}".format(w, b))

    # predict some numbers
    x_pred = np.array([[1,6,5],[2,7,3],[3,8,-2],[-4,-9,0],[5,-10,1]])
    x_pred = x_pred / 10.0
    y_pred = np.argmax(forward(x_pred, w, b), axis=1)
    print("Predictions: {0}".format(y_pred))

if __name__ == '__main__':
    main()