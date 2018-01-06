import common
import numpy as np
import tensorflow as tf
import pandas as pd
import math
import matplotlib.pyplot as plt

from common import Mode

#MODE = Mode.TRAIN
MODE = Mode.EVAL
#MODE = Mode.PREDICT

NUM_FEATURES = 784
NUM_EPOCHS = 10000
LEARNING_RATE = 0.001
BATCH_SIZE = 32

def main():
    tf.logging.set_verbosity(tf.logging.INFO)

    common.download_data() # download mnist dataset if it does not exist yet

    # NN
    input = tf.placeholder(tf.float32, shape=(None, NUM_FEATURES))
    hidden = tf.layers.dense(input, 100, activation=tf.nn.sigmoid)
    recon = tf.layers.dense(hidden, NUM_FEATURES, activation=tf.nn.sigmoid)

    l2_loss = tf.reduce_sum(tf.squared_difference(input, recon), axis=1)
    mean_loss = tf.reduce_mean(l2_loss)

    optimize = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(mean_loss)

    saver = tf.train.Saver() # ops for saving

    with tf.Session() as sess:
        if MODE == Mode.TRAIN: # train model
            df = pd.read_csv("./mnist/mnist_train.csv")

            samp = df.iloc[:,1:].as_matrix() # features
            samp = samp.astype(float)
            samp = samp / 255.0 # normalize to [0,1]
            sample_size = samp.shape[0]

            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

            for i in range(0, NUM_EPOCHS):
                for j in range(0, math.ceil(float(sample_size)/BATCH_SIZE)):
                    sess.run([optimize], feed_dict={input: samp[j*BATCH_SIZE:(j+1)*BATCH_SIZE, :]})

                l = sess.run([mean_loss], feed_dict={input: samp})
                print("Epoch {0}: Loss {1}".format(i,l))

                # save model checkpoint each epoch
                saver.save(sess, "./checkpoints/vae", global_step=i)

            print("Done training. Saving weights to disk")

        elif MODE == Mode.EVAL: # try to compress and reconstruct a MNIST datapoint
            df = pd.read_csv("./mnist/mnist_test.csv")

            samp = df.iloc[:,1:].as_matrix() # features
            samp = samp.astype(float)
            samp = samp / 255.0 # normalize to [0,1]

            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

            checkpoint_path = tf.train.latest_checkpoint("./checkpoints")
            saver.restore(sess, checkpoint_path)

            ml, l2, r = sess.run([mean_loss, l2_loss, recon], feed_dict={input: samp})
            print("Mean evaluation loss: {0}".format(ml))

            r = np.reshape(r, (-1, 28, 28))
            samp = np.reshape(samp, (-1, 28, 28))

            worst_idx = np.argmax(l2)
            best_idx = np.argmin(l2)

            # plot original and reconstruction for best and worst examples in training dataset
            plt.subplot(2,2,1)
            plt.imshow(samp[best_idx, :, :], cmap="gray")

            plt.subplot(2,2,2)
            plt.imshow(r[best_idx, :, :], cmap="gray")

            plt.subplot(2,2,3)
            plt.imshow(samp[worst_idx, :, :], cmap="gray")

            plt.subplot(2,2,4)
            plt.imshow(r[worst_idx, :, :], cmap="gray")
            plt.show()
        
if __name__ == '__main__':
    main()

