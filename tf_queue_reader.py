import common
import tensorflow as tf

NUM_CLASSES = 10
NUM_FEATURES = 784

NUM_EXAMPLES = 10000

NUM_EPOCHS = 1

LEARNING_RATE = 0.0001

BATCH_SIZE = 32
DISPLAY_FREQUENCY = 500

def main():
    # Check that we have correct TensorFlow version installed
    tf_version = tf.__version__
    print("TensorFlow version: {}".format(tf_version))
    assert "1.4" <= tf_version, "TensorFlow r1.4 or later is needed"

    tf.logging.set_verbosity(tf.logging.INFO)

    common.download_data() # download mnist dataset if it does not exist yet

    # input pipeline
    csv_file = tf.train.string_input_producer(["./mnist/mnist_train.csv"], num_epochs=NUM_EPOCHS)
    line_reader = tf.TextLineReader()
    _, csv_row = line_reader.read(csv_file)

    record_defaults = [[0] for row in range(0,NUM_FEATURES+1)]
    data = tf.decode_csv(csv_row,record_defaults=record_defaults)

    label = tf.to_float(tf.one_hot(data[0], NUM_CLASSES))
    feature = tf.to_float(tf.reshape(data[1:], (28,28)))

    feature_batch, label_batch = tf.train.batch([feature, label], batch_size=BATCH_SIZE)

    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        num_steps = 0

        try:
            while not coord.should_stop():
                f, l = sess.run([feature_batch, label_batch])
                num_steps += 1

                if num_steps % DISPLAY_FREQUENCY == 0:
                    print("Read {0} batches of data...".format(num_steps))

        except tf.errors.OutOfRangeError:
            print("Done reading {0} batches during {1} epochs".format(num_steps, NUM_EPOCHS))
        finally:
            coord.request_stop()

        coord.join(threads)

if __name__ == '__main__':
    main()

