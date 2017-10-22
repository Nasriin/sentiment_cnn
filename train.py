import data_helpers
import tensorflow as tf
import numpy as np
from tensorflow.contrib.keras.python.keras.datasets import imdb
from cnn import CNN


print("Loading data...")
top_words = 5000 # Collection frequency
num_classes = 2
num_filters = 15
max_len = 100 # check the data_analysis.ipynb for a box plot.
embedding_size = 10
batch_size = 100
epocs = 10

# Load data
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words= top_words)

padded_x_train = []
for x in x_train:
    if len(x) < max_len:
        x += [0] * (max_len - len(x))
    x = x[:max_len]
    padded_x_train.append(x)

expanded_y = []
for y in y_train:
    if y == 0:
        expanded_y.append([0, 1])
    else:
        expanded_y.append([1, 0])

y_train = np.array(expanded_y)
x_train = np.array(padded_x_train)

input_x = tf.placeholder(tf.int32, [None, max_len], name="input_x")
input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")

# Training
embedding = tf.Variable(tf.random_uniform([top_words, embedding_size], -1.0, 1.0))
embedded_input_x = tf.nn.embedding_lookup(embedding, input_x)
print(embedded_input_x.get_shape())
embedded_input_x = tf.reshape(embedded_input_x, [-1, max_len, 1, embedding_size])
print(embedded_input_x.get_shape())

# Convolutional network

# Convolutional Layer #1
conv1 = tf.layers.conv2d(embedded_input_x, num_filters, [5, embedding_size], padding = "same")
print(conv1.get_shape())

# Pooling Layer #1
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[max_len, 1], strides=1)
print(pool1.get_shape())
pool1 = tf.contrib.layers.flatten(pool1)

logits = tf.layers.dense(pool1, units=num_classes)
prediction = tf.nn.softmax(logits)

loss = tf.nn.softmax_cross_entropy_with_logits(labels=input_y, logits=logits)
optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(loss)

promt = 100
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for ep in range(epocs):
        avg_loss = 0.0
        for i in range(x_train.shape[0] // batch_size):
            start_index = i * batch_size
            feed_dict = {
                input_x: x_train[start_index:start_index+batch_size],
                input_y: y_train[start_index:start_index+batch_size]
            }
            fetches = [loss, train_op]
            l, _ = sess.run(fetches, feed_dict)
            avg_loss += np.average(l)
            if i % promt == 0:
                print(avg_loss/promt)
                avg_loss = 0.0


if __name__ == '__main__':
    print('start')