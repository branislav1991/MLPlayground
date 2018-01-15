import common
import numpy as np
import tensorflow as tf
import pandas as pd
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from common import Mode

#MODE = Mode.TRAIN
#MODE = Mode.EVAL
MODE = Mode.PREDICT

NUM_FEATURES = 784
NUM_HIDDEN_FEATURES_1 = 16
NUM_HIDDEN_FEATURES_2 = 32
NUM_LATENT_FEATURES = 100
NUM_EPOCHS = 200
LEARNING_RATE = 0.005
BATCH_SIZE = 32

class VAE():
    def __init__(self):
        """Initializes the vae. Configures the tensorflow graph.
        """
        # configure NN model
        input = tf.placeholder(tf.float32, shape=(None, 28, 28, 1), name="input")
        self.z_mean, self.z_stddev = self.__encoder(input)

        # random sampling from a unit normal distribution
        samples = tf.random_normal([tf.shape(self.z_stddev)[0], NUM_LATENT_FEATURES], 0, 1, dtype=tf.float32)
        self.sampled_z = tf.add(self.z_mean, (self.z_stddev * samples), name="sampled_z")

        self.recon = self.__decoder(self.sampled_z)

        # const functions
        input_flat = tf.layers.flatten(input)
        recon_flat = tf.layers.flatten(self.recon)

        KL_factor = tf.placeholder(tf.float32, name="KL_factor")

        with tf.variable_scope("loss"):
            self.l2_loss = tf.reduce_sum(tf.squared_difference(input_flat, recon_flat), axis=1)
            self.KL_loss = 0.5 * tf.reduce_sum(tf.square(self.z_mean) + tf.square(self.z_stddev) - tf.log(tf.square(self.z_stddev)) - 1, axis=1) # KL-divergence between 2 Gaussians
            self.mean_loss = tf.reduce_mean(self.l2_loss + KL_factor * self.KL_loss)

        with tf.variable_scope("optimizer"):
            #self.optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(self.mean_loss)
            self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.mean_loss)

        self.tbmerge = tf.summary.merge_all() # tensorboard visualization

    def __encoder(self, input):
        """The encoder part of the vae.
        Returns: z_mean and z_stddev layers
        """
        with tf.variable_scope("encoder"):
            h1 = tf.layers.conv2d(input, NUM_HIDDEN_FEATURES_1, [3,3], activation=tf.nn.relu, name="hidden_1")
            h2 = tf.layers.conv2d(h1, NUM_HIDDEN_FEATURES_2, [3,3], activation=tf.nn.relu, name="hidden_2")
            h3 = tf.layers.flatten(h1, name="hidden_3")

            z_mean = tf.layers.dense(h3, NUM_LATENT_FEATURES, name="z_mean")
            tf.summary.histogram("z_mean", z_mean)
            z_stddev = tf.layers.dense(h3, NUM_LATENT_FEATURES, name="z_stddev")
            tf.summary.histogram("z_stddev", z_stddev)
        
        return z_mean, z_stddev

    def __decoder(self, hidden):
        """The decoder (reverse) part of the vae.
        Returns: final layer
        """
        with tf.variable_scope("decoder"):
            h1 = tf.layers.dense(hidden, NUM_HIDDEN_FEATURES_2*24*24, activation=tf.nn.relu, name="hidden_1")
            h2 = tf.reshape(h1, [-1,24,24,NUM_HIDDEN_FEATURES_2], name="hidden_2")
            h3 = tf.layers.conv2d_transpose(h2, 1, [3,3], activation=tf.nn.relu, name="hidden_3")
            out = tf.layers.conv2d_transpose(h3, 1, [3,3], activation=tf.nn.sigmoid, name="reconstruction")

        return out

    def train(self, features, batch_size, learning_rate, num_epochs, display_rate=5, warmup=10):
        """Trains the vae for num_epochs epochs using a certain batch size and learning rate.
        Also Displays mean loss every display_rate epochs.
        Uses a warmup scheme for KL loss to improve the L2 loss. During warmup, the KL loss term is weighted
        gradually from 0 to 1.
        Returns: final z_mean, z_stddev evaluated for input data
        """
        dataset_size = features.shape[0]
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            train_writer = tf.summary.FileWriter(".tensorboard/vae/train", sess.graph)
            num_batches = math.ceil(float(dataset_size)/batch_size)

            for i in range(0, num_epochs):
                KL_factor = i/warmup if i < warmup else 1.0
                for j in range(0, num_batches):
                    summary,_ = sess.run([self.tbmerge, self.optimizer], 
                        feed_dict={"input:0": features[j*batch_size:(j+1)*batch_size, :], "KL_factor:0": KL_factor})
                    train_writer.add_summary(summary, num_batches*i + j)

                if i % display_rate == 0:
                    kl, l2 = sess.run([self.KL_loss, self.l2_loss], feed_dict={"input:0": features})
                    l = sess.run([self.mean_loss], feed_dict={"input:0": features, "KL_factor:0": KL_factor})
                    print("Epoch {0}: Loss {1} with KL loss factor {2}".format(i,l, KL_factor))

                # save model checkpoint each epoch
                saver.save(sess, "./checkpoints/vae/vae", global_step=i)

            print("Done training.")

            z_mean_eval, z_stddev_eval = sess.run([self.z_mean, self.z_stddev], feed_dict={"input:0": features})
            return z_mean_eval, z_stddev_eval

    def evaluate(self, features):
        """Evaluates the reconstruction ability of the network.
        Returns: mean l2 loss, reconstructed digits
        """
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

            checkpoint_path = tf.train.latest_checkpoint("./checkpoints/vae")
            saver.restore(sess, checkpoint_path)

            l2, r = sess.run([self.l2_loss, self.recon], feed_dict={"input:0": features})

            r = np.reshape(r, (-1, 28, 28))
            orig = np.reshape(features, (-1, 28, 28))

        return l2, r

    def generate(self, latent):
        """Generates a random hand-written digit using the provided latent vector 
        representation. Requires a trained vae.
        Returns: generated digit
        """
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

            checkpoint_path = tf.train.latest_checkpoint("./checkpoints/vae")
            saver.restore(sess, checkpoint_path)

            gen = sess.run(self.recon, feed_dict={"sampled_z:0": latent})
            gen = np.reshape(gen, (-1, 28, 28))

        return gen

def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    common.download_mnist_data()
    common.create_checkpoint_folder()
    
    vae = VAE() # initialize variational encoder model

    if MODE == Mode.TRAIN:
        df = pd.read_csv("./mnist/mnist_train.csv")
        samp = df.iloc[:1000,1:].astype("float64").divide(255.0).as_matrix() # features
        classes = df.iloc[:1000,0].as_matrix() # features

        samp = np.reshape(samp, [-1,28,28,1])

        z_mean_eval, z_stddev_eval = vae.train(features=samp, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, num_epochs=NUM_EPOCHS, warmup=10)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.scatter(z_mean_eval[:,0], z_mean_eval[:,1], z_mean_eval[:,2], c=classes)
        plt.show()

    elif MODE == Mode.EVAL:
        df = pd.read_csv("./mnist/mnist_test.csv")
        samp = df.iloc[:,1:].astype("float64").divide(255.0).as_matrix() # features
        samp = np.reshape(samp, [-1,28,28,1])

        l2_loss, recon = vae.evaluate(features=samp)

        mean_loss = np.mean(l2_loss, axis=0)
        print("Mean evaluation loss: {0}".format(mean_loss))

        worst_idx = np.argmax(l2_loss)
        best_idx = np.argmin(l2_loss)
        median_idx = np.where(l2_loss == np.percentile(l2_loss, 50, interpolation="nearest"))[0][0]

        # plot original and reconstruction for best and worst examples in training dataset
        common.plot_sub(samp[best_idx, :, :, 0], "Best (sample)", 1)
        common.plot_sub(recon[best_idx, :, :], "Best (recon)", 2)

        common.plot_sub(samp[median_idx, :, :, 0], "Median (sample)", 3)
        common.plot_sub(recon[median_idx, :, :], "Median (recon)", 4)

        common.plot_sub(samp[worst_idx, :, :, 0], "Worst (sample)", 5)
        common.plot_sub(recon[worst_idx, :, :], "Worst (recon)", 6)
        plt.show()

    elif MODE == Mode.PREDICT:
        z = np.zeros((6,100)) # some latent vector
        z[0,0:60] = 1.0
        z[1,10:70] = 1.0
        z[2,20:80] = 1.0
        z[3,30:90] = 1.0
        z[4,40:100] = 1.0

        gen = vae.generate(z)

        common.plot_sub(gen[0, :, :], "Gen 1", 1)
        common.plot_sub(gen[1, :, :], "Gen 2", 2)

        common.plot_sub(gen[2, :, :], "Gen 3", 3)
        common.plot_sub(gen[3, :, :], "Gen 4", 4)

        common.plot_sub(gen[4, :, :], "Gen 5", 5)
        common.plot_sub(gen[5, :, :], "Gen 6", 6)
        plt.show()
        
if __name__ == '__main__':
    main()

