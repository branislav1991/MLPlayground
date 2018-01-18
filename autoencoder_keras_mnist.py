import os
import common
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Flatten, Reshape
from keras.callbacks import ModelCheckpoint
import pandas as pd
import matplotlib.pyplot as plt

from common import Mode

MODE = Mode.TRAIN
#MODE = Mode.EVAL
#MODE = Mode.PREDICT

NUM_FEATURES = 784
NUM_HIDDEN_FEATURES = 100
NUM_EPOCHS = 100
LEARNING_RATE = 0.0005
BATCH_SIZE = 32

def create_keras_ae_checkpoint_folder():
    if not os.path.exists(common.CHECKPOINT_DIR + "/keras_ae"):
        os.makedirs(common.CHECKPOINT_DIR + "/keras_ae")

def main():
    print("Downloading data and setting up evironment...")
    common.download_mnist_data() # download mnist dataset if it does not exist yet
    common.create_checkpoint_folder()

    # NN
    print("Configuring neural network...")
    input = Input(shape=(28,28,1))
    hidden_1 = Conv2D(32,(3,3), activation='relu')(input)
    hidden_2 = Conv2D(1,(3,3), activation='relu')(hidden_1)
    hidden_3 = Flatten()(hidden_2)
    hidden_4 = Dense(NUM_HIDDEN_FEATURES, activation='sigmoid')(hidden_3)

    hidden_5 = Dense(24*24, activation='relu')(hidden_4)
    hidden_6 = Reshape((24,24,1))(hidden_5)
    hidden_7 = Conv2DTranspose(32, (3,3), activation='relu')(hidden_6)
    output = Conv2DTranspose(1, (3,3), activation='sigmoid')(hidden_7)

    print("Compiling model...")
    model = Model(inputs=input, outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error')

    if MODE == Mode.TRAIN:
        create_keras_ae_checkpoint_folder()

        print("Loading training data from hard disk...")
        # Load data and run sgd algorithm
        df = pd.read_csv("./mnist/mnist_train.csv")

        samp = df.iloc[:,1:].astype("float64").divide(255.0).as_matrix() # features
        sample_size = samp.shape[0]

        samp = np.reshape(samp, (-1,28,28,1))

        print("Training...")
        callback = ModelCheckpoint("./checkpoints/keras_ae/model.{epoch:02d}-{loss:.2f}.hdf5", monitor='loss', verbose=0, save_best_only=False, save_weights_only=True)
        model.fit(samp, samp, batch_size=32, epochs=NUM_EPOCHS, callbacks=[callback])

        model.save_weights("./checkpoints/keras_ae/model.final.hdf5")

        print("Done training.")

    elif MODE == Mode.EVAL: # try to compress and reconstruct a MNIST datapoint
        df = pd.read_csv("./mnist/mnist_test.csv")

        samp = df.iloc[:,1:].astype("float64").divide(255.0).as_matrix() # features
        samp = np.reshape(samp, (-1,28,28,1))

        print("Evaluating...")
        model.load_weights("./checkpoints/keras_ae/model.final.hdf5")

        l2 = np.zeros((samp.shape[0],))
        r = np.zeros(samp.shape)
        for i in range(0, samp.shape[0]):
            l2[i] = model.test_on_batch(samp[None,i,:,:,:], samp[None,i,:,:,:])
            r[i,:,:,:] = model.predict_on_batch(samp[None,i,:])
            
        ml = np.mean(l2)
        print("Mean evaluation loss: {0}".format(ml))

        print("Done evaluating.")

        r = np.reshape(r, (-1, 28, 28))
        samp = np.reshape(samp, (-1, 28, 28))

        worst_idx = np.argmax(l2)
        best_idx = np.argmin(l2)
        median_idx = np.where(l2 == np.percentile(l2, 50, interpolation='nearest'))[0][0]

        # plot original and reconstruction for best and worst examples in training dataset
        common.plot_sub(samp[best_idx, :, :], "Best (sample)", 1)
        common.plot_sub(r[best_idx, :, :], "Best (recon)", 2)

        common.plot_sub(samp[median_idx, :, :], "Median (sample)", 3)
        common.plot_sub(r[median_idx, :, :], "Median (recon)", 4)

        common.plot_sub(samp[worst_idx, :, :], "Worst (sample)", 5)
        common.plot_sub(r[worst_idx, :, :], "Worst (recon)", 6)

        plt.show()
        
if __name__ == '__main__':
    main()

