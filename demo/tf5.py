from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 3000
MOVING_AVERAGE_DECAY = 0.99


def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    if avg_class is None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)


def train(mnist):
    with tf.Session() as sess:
        x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
        weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
        biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
        weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
        biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))
        y = inference(x, None, weights1, biases1, weights2, biases2)
        global_step = tf.Variable(0, trainable=False)
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        tf.initialize_all_variables().run()
        variable_averages_op = variable_averages.apply(tf.trainable_variables())
        average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_, 1), logits=y)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
        regularization = regularizer(weights1) + regularizer(weights2)
        loss = cross_entropy_mean + regularization
        learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE,
            global_step,
            mnist.train.num_examples/BATCH_SIZE,
            LEARNING_RATE_DECAY
        )
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
        with tf.control_dependencies([train_step, variable_averages_op]):
            train_op = tf.no_op(name='train')
            correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            init = tf.initialize_all_variables()
            sess.run(init)
            validate_feed = {
                    x: mnist.validation.images,
                    y_: mnist.validation.labels
                }
            test_feed = {
                    x: mnist.test.images,
                    y_: mnist.test.labels
                }
            for i in range(TRAINING_STEPS):
                if i % 100 == 0:
                    validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                    print('after %d training step(s), validation accuracy using average model is %g' % (i, validate_acc))
                xs, ys = mnist.train.next_batch(BATCH_SIZE)
                sess.run(train_op, feed_dict={x: xs, y_: ys})
            test_acc = sess.run(accuracy, feed_dict=test_feed)
            print('after %d training step(s), test accuracy using average model is %g' % (TRAINING_STEPS, test_acc))


def main(argv=None):
    mnist = input_data.read_data_sets('./data', one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()


# print('training data size:', mnist.train.num_examples)
# print('validating data size:', mnist.validation.num_examples)
# print('testing data size:', mnist.test.num_examples)
# # print('example training data:', mnist.train.images[0])
# print('example training data label:', mnist.train.labels[0])
#
# xs, ys = mnist.train.next_batch(batch_size)
# print("X shape:", xs.shape)
# print("Y shape: ", ys.shape)


