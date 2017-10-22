import data_helpers
import tensorflow as tf
from cnn import CNN

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Data Preparation
# ==================================================

# Load data
print("Loading data...")
x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)

# Training
# ==================================================

# cross_entropy = tf.reduce_mean(
#     tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    cnn = CNN(
        sequence_length=x_train.shape[1],
        num_classes=y_train.shape[1],
        # vocab_size=len(vocab_processor.vocabulary_),
        # embedding_size=FLAGS.embedding_dim,
        # filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
        # num_filters=FLAGS.num_filters,
        # l2_reg_lambda=FLAGS.l2_reg_lambda
    )

    # Define Training procedure
    # global_step = tf.Variable(0, name="global_step", trainable=False)
    # optimizer = tf.train.AdamOptimizer(1e-3)
    # grads_and_vars = optimizer.compute_gradients(cnn.loss)
    # train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    sess.run(tf.global_variables_initializer())
    # Training loop.
    for i in range(20000):
    batch = mnist.train.next_batch(50)

    if i % 100 == 0:
      train_accuracy = accuracy.eval(feed_dict={
          x: batch[0], y_: batch[1], keep_prob: 1.0})
      print('step %d, training accuracy %g' % (i, train_accuracy))

train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

  print('test accuracy %g' % accuracy.eval(feed_dict={
      x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
