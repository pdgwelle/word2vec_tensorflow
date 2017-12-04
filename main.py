import numpy as np
import pandas as pd

import tensorflow as tf

##
vocabulary_size = 
embedding_size = 

## initialize word embeddings
embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

## initialize model weights
nce_weights = tf.Variable(
  tf.truncated_normal([vocabulary_size, embedding_size],
                      stddev=1.0 / math.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

## context and target words, represented as integers
train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

## look up vectorfrom train_inputs (context) in the embeddings
embed = tf.nn.embedding_lookup(embeddings, train_inputs)

## loss function
loss = tf.reduce_mean(
  tf.nn.nce_loss(weights=nce_weights,
                 biases=nce_biases,
                 labels=train_labels,
                 inputs=embed,
                 num_sampled=num_sampled,
                 num_classes=vocabulary_size)
  	)

## solver
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)

## run session
for inputs, labels in generate_batch(...):
  feed_dict = {train_inputs: inputs, train_labels: labels}
  _, cur_loss = session.run([optimizer, loss], feed_dict=feed_dict)

def generate_batch():









### Tutorial can be found here:
### https://www.tensorflow.org/tutorials/word2vec