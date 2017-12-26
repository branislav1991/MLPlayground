"""Performs logistic regression using gradient descent.

In this version of logistic regression, we use the MNIST
dataset and train on minibatches of the dataset"""

import numpy as np
import pandas as pd
import math
import common

NUM_CLASSES = 10
NUM_FEATURES = 784

NUM_EXAMPLES = 1000

NUM_EPOCHS = 10000

LEARNING_RATE = 0.01
LOSS_DISPLAY_FREQUENCY = 100

BATCH_SIZE = 32

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

def one_hot(labels, num_classes): # convert label numbers to one-hot encoding
    y = [[1 if y==x else 0 for y in range(0, num_classes)] for x in labels]
    y = np.array(y)
    return y

def forward(x, w, b): # forward pass through network
    return x.dot(w) + b

def main():
    common.download_data() # download mnist dataset if it does not exist yet
    df = pd.read_csv("./mnist/mnist_train.csv")

    x = df.iloc[0:NUM_EXAMPLES,1:].as_matrix() # features
    x = x.astype(float)
    x = x / 255.0 # normalize to [0,1]

    y = df.iloc[0:NUM_EXAMPLES,0].as_matrix() #labels
    y = one_hot(y, NUM_CLASSES)

    w = np.random.rand(NUM_FEATURES, NUM_CLASSES) # randomly initialize weights
    b = np.zeros(NUM_CLASSES) # initialize biases to zero

    # perform batch gradient descent
    for i in range(0, NUM_EPOCHS):
        for j in range(0, math.ceil(float(NUM_EXAMPLES) / float(BATCH_SIZE))):
            x_ex = x[j*BATCH_SIZE:(j+1)*BATCH_SIZE,:]
            y_ex = y[j*BATCH_SIZE:(j+1)*BATCH_SIZE,:]

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

    # evaluate on the test set
    df = pd.read_csv("./mnist/mnist_test.csv")
    x_pred = df.iloc[:,1:].as_matrix() # features
    x_pred = x_pred.astype(float)
    x_pred = x_pred / 255.0 # normalize to [0,1]

    y_eval = df.iloc[:,0].as_matrix() #labels
    y_eval = one_hot(y_eval, 10)

    y_pred = forward(x_pred, w, b)
    print("Evaluation accuracy: {0}".format(accuracy(y_pred, y_eval)))

if __name__ == '__main__':
    main()