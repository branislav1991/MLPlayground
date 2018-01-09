import common
import numpy as np
import tensorflow as tf
import pandas as pd
import math

NUM_CLASSES = 10
NUM_FEATURES = 784

NUM_EXAMPLES = 10000

NUM_EPOCHS = 10000

LEARNING_RATE = 0.0001

BATCH_SIZE = 32

def main():
    tf.logging.set_verbosity(tf.logging.INFO)

    common.download_mnist_data() # download mnist dataset if it does not exist yet
    df = pd.read_csv("./mnist/mnist_train.csv")

    x_samp = df.iloc[0:NUM_EXAMPLES,1:].as_matrix() # features
    x_samp = x_samp.astype(float)
    x_samp = x_samp / 255.0 # normalize to [0,1]

    y_samp = df.iloc[0:NUM_EXAMPLES,0].as_matrix() #labels
    y_samp = tf.one_hot(y_samp, NUM_CLASSES)

    # NN
    x = tf.placeholder(tf.float32, shape=(None, NUM_FEATURES))
    dense_1 = tf.layers.dense(x, 100, activation=tf.nn.relu)
    pred = tf.layers.dense(dense_1, NUM_CLASSES, activation=tf.nn.relu)

    y = tf.placeholder(tf.float32, shape=(None, NUM_CLASSES))

    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

    correct_prediction = tf.equal(tf.argmax(pred,axis=1), tf.argmax(y,axis=1))
    accuracy = tf.reduce_mean(tf.to_float(correct_prediction))

    optimize = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        y_one_hot = sess.run(y_samp)

        for i in range(0, NUM_EPOCHS):
            for j in range(0, math.ceil(float(NUM_EXAMPLES)/BATCH_SIZE)):
                sess.run([optimize], feed_dict={x: x_samp[j*BATCH_SIZE:(j+1)*BATCH_SIZE, :], y: y_one_hot[j*BATCH_SIZE:(j+1)*BATCH_SIZE, :]})

            l, a = sess.run([loss, accuracy], feed_dict={x: x_samp, y: y_one_hot})
            print("Epoch {0}: Loss {1}, Accuracy {2}".format(i,l,a))

    print("Done training")

if __name__ == '__main__':
    main()

