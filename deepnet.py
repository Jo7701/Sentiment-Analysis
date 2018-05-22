import tensorflow as tf
import cPickle as pickle
import numpy as np
import sys
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.python.ops import rnn, rnn_cell

print 'Loading Data'
with open('train_test_data.pickle', 'rb') as file:
	train_x, train_y, test_x, test_y = pickle.load(file)

rnn_size = 128
chunk_size = 9
n_chunks = 47
n_classes = 2
batch_size = 100

x = tf.placeholder('float', [None, len(train_x[0])])
y = tf.placeholder('float')

def neural_network(x):
	layer = {'weights': tf.Variable(tf.random_normal([rnn_size, n_classes])), 'biases': tf.Variable(tf.random_normal([n_classes]))}

	x = tf.reshape(x, (-1, n_chunks, chunk_size))
	x = tf.transpose(x, [1,0,2])
	x = tf.reshape(x, (-1, chunk_size))
	x = tf.split(x, n_chunks)

	lstm = rnn_cell.BasicLSTMCell(rnn_size)
	outputs, states = rnn.static_rnn(lstm, x, dtype = tf.float32)

	return tf.matmul(outputs[-1], layer['weights']) + layer['biases']

def user_input():
	input_data = raw_input("Enter a sentence: ")
	with open('lexicon.pickle', 'rb') as file:
		lexicon = pickle.load(file)
	lemmatizer = WordNetLemmatizer()
	sentence = word_tokenize(input_data)
	sentence = [lemmatizer.lemmatize(i) for i in sentence]

	toRet = np.zeros(len(lexicon))
	for word in sentence:
		if word in lexicon:
			toRet[lexicon.index(word)] += 1
	return np.reshape(toRet, (1, len(lexicon)))


def train(x):
	prediction = neural_network(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y))
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		n_epochs = 10
		
		for epoch in range(n_epochs):
			loss = 0
			
			i = 0
			while(i < len(train_x)):
				start = i
				end = i + batch_size

				epoch_x = np.array(train_x[start:end])
				epoch_y = np.array(train_y[start:end])

				_, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
				loss += c
				i += batch_size

			print 'Epoch ', epoch + 1, ' Loss: ', loss

		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print 'Accuracy: ', accuracy.eval(feed_dict = {x: test_x, y: test_y})

		vectorized = user_input()
		output = prediction.eval(feed_dict = {x:vectorized})
		print output

		if tf.argmax(output,1).eval() == 0:
			print 'The sentence is positive'
		else:
			print 'The sentence is negative'

train(x)
