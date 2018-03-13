import tensorflow as tf
import numpy as np
from ops import *



# Reference : https://github.com/dennybritz/cnn-text-classification-tf/blob/master/text_cnn.py

class TextCNN(object):
	"""
	A CNN for text classification.
	Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
	"""
	def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, num_filters, lambda_l2):
		self.name = 'TextCNN'
		print(self.name)
		self.sequence_length = sequence_length # 21
		self.num_classes = num_classes # 13 # ['anger', 'boredom', 'empty', 'enthusiasm', 'fun', 'happiness', 'hate', 'love', 'neutral', 'relief', 'sadness', 'surprise', 'worry']
		self.vocab_size = vocab_size # 5567
		self.embedding_size = embedding_size
		self.num_filters = num_filters # 128	
		self.filter_size = [1,2,3,4,5]
		self.dense_size = [512, 256]
		#self.dropout_keep_prob = dropout_keep_prob
		# Keeping track of l2 regularization loss (optional)
		self.l2_loss = tf.constant(0.0)
		self.lambda_l2 = lambda_l2
		self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
		self.input_y = tf.placeholder(tf.int64, [None, ], name="input_y")
		self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
		self.pretrained_embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], dtype=tf.float32, minval=-1.0, maxval=1.0), trainable=True, name='pretrained_embeddings')
		# specify some class weightings
		self.class_weights = tf.constant([1.0, 1.0])
		#self.test_building(self.input_x)
		self.build_network()
		with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
			self.train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.6, beta2=0.9).minimize(self.loss)
	def build_network(self):
		with tf.variable_scope(self.name):
			with tf.variable_scope('Embedding'):
				self.embedded_chars = tf.nn.embedding_lookup(self.pretrained_embeddings, self.input_x)
				self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
				print('embedded_chars_expanded.shape : ', self.embedded_chars_expanded.shape)
			pooled_outputs = []
			for index, filter_size in enumerate(self.filter_size):
				with tf.variable_scope('Conv' + str(index)):
					W_conv = weight_variables([filter_size, self.embedding_size, 1, self.num_filters], name = 'W_conv' + str(index))
					b_conv = bias_variables([self.num_filters], name = 'b_conv' + str(index))
					h_conv = conv2d(self.embedded_chars_expanded, W_conv, 1, 'VALID') + b_conv
					conv = lrelu(h_conv) # shape : vocab_size,embedding_size,128
					print('conv shape : ', conv.shape)
					pooled = max_pool(conv, self.sequence_length - filter_size + 1, 1, 'VALID')
					pooled = dropout(pooled, self.dropout_keep_prob)
					#pooled = max_pool(pooled, 1, self.num_filters, 'VALID')
					pooled_outputs.append(pooled)
					print('pooled shape : ', pooled.shape)

			# Combine all the pooled features
			num_filters_total = self.num_filters * len(self.filter_size) # 128, 128, 128
			self.h_pool = tf.concat(pooled_outputs, 3)
			self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
			print(self.h_pool_flat.shape)

			# Add dropout
			with tf.name_scope("dropout"):
				self.h_drop = dropout(self.h_pool_flat, self.dropout_keep_prob)
			with tf.name_scope("output"):
				W_output = weight_variables([num_filters_total, self.num_classes], name = 'W_output')
				b_output = bias_variables([self.num_classes], name = 'b_output')
				self.l2_loss += tf.nn.l2_loss(W_output)
				self.l2_loss += tf.nn.l2_loss(b_output)
				# dense = dropout(relu(tf.matmul(self.h_drop, W_output) + b_output), self.dropout_keep_prob)

				# W_output1 = weight_variables([self.dense_size[0], self.dense_size[1]], name = 'W_output1')
				# b_output1 = bias_variables([self.dense_size[1]], name = 'b_output1')
				# self.l2_loss += tf.nn.l2_loss(W_output1)
				# self.l2_loss += tf.nn.l2_loss(b_output1)
				# dense1 = dropout(relu(tf.matmul(dense, W_output1) + b_output1), self.dropout_keep_prob)

				# W_output2 = weight_variables([self.dense_size[1], self.num_classes], name = 'W_output2')
				# b_output2 = bias_variables([self.num_classes], name = 'b_output2')
				# self.l2_loss += tf.nn.l2_loss(W_output2)
				# self.l2_loss += tf.nn.l2_loss(b_output2)
				# self.scores = tf.matmul(dense1, W_output2) + b_output2

				self.scores = tf.matmul(self.h_drop, W_output) + b_output
				self.predictions = tf.argmax(tf.nn.softmax(self.scores), 1, name="predictions")

			# Calculate mean cross-entropy loss
			with tf.name_scope("loss"):
				print('score dim : ', self.scores.shape)
				print('input_y dim : ', self.input_y.shape)
				balanced_weight = tf.gather(self.class_weights, self.input_y)
				losses = tf.losses.sparse_softmax_cross_entropy(self.input_y, self.scores, weights=balanced_weight)
				self.loss = tf.reduce_mean(losses) + self.lambda_l2*self.l2_loss
			# Accuracy
			with tf.name_scope("accuracy"):
				correct_predictions = tf.equal(self.predictions, self.input_y)
				self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


	'''			
	def __call__(self, input_x, input_y):
		with tf.variable_scope(self.name):
			with tf.variable_scope('Embedding'):
				self.W = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0), name="W")
				self.embedded_chars = tf.nn.embedding_lookup(self.W, input_x)
				self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
				print('embedded_chars_expanded.shape : ', self.embedded_chars_expanded.shape)
			pooled_outputs = []
			for index, filter_size in enumerate(self.filter_size):
				with tf.variable_scope('Conv' + str(index)):
					W_conv = weight_variables([filter_size, self.embedding_size, 1, self.num_filters], name = 'W_conv' + str(index))
					b_conv = bias_variables([self.num_filters], name = 'b_conv' + str(index))
					h_conv = conv2d(self.embedded_chars_expanded, W_conv, 1, 'VALID') + b_conv
					conv = relu(h_conv) # shape : vocab_size,embedding_size,128
					print('conv shape : ', conv.shape)
					pooled = max_pool(conv, self.sequence_length - filter_size + 1, 1, 'VALID')
					#pooled = max_pool(pooled, 1, self.num_filters, 'VALID')
					pooled_outputs.append(pooled)
					print('pooled shape : ', pooled.shape)

			# Combine all the pooled features
			num_filters_total = self.num_filters * len(self.filter_size) # 128, 128, 128
			self.h_pool = tf.concat(pooled_outputs, 3)
			self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
			print(self.h_pool_flat.shape)

			# Add dropout
			with tf.name_scope("dropout"):
				self.h_drop = dropout(self.h_pool_flat, self.dropout_keep_prob)
			with tf.name_scope("output"):
				W_output = weight_variables([len(self.filter_size), self.num_classes], name = 'W_output')
				b_output = bias_variables([self.num_classes], name = 'b_output')
				self.l2_loss += tf.nn.l2_loss(W_output)
				self.l2_loss += tf.nn.l2_loss(b_output)
				self.scores = tf.matmul(self.h_drop, W_output) + b_output
				self.predictions = tf.argmax(self.scores, 1, name="predictions")

			# Calculate mean cross-entropy loss
			with tf.name_scope("loss"):
				losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=input_y)
				self.loss = tf.reduce_mean(losses) + self.lambda_l2*self.l2_loss
			# Accuracy
			with tf.name_scope("accuracy"):
				correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
				self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
			with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
				self.train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.6, beta2=0.9).minimize(self.loss)
	'''
#network = TextCNN(20,10,500,100,128)






