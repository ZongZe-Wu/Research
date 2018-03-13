import tensorflow as tf 
from ops import *

class Text_Embedding(object):
	def __init__(self, args, name, vocab_size, embedding_size):
		self.args = args
		self.name = name
		self.vocab_size = vocab_size
		self.embedding_size = embedding_size
		self.pretrained_embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], dtype=tf.float32, minval=-1.0, maxval=1.0), trainable=True, name='pretrained_embeddings')
	def __call__(self, input_x):
		with tf.variable_scope(self.name):
			with tf.variable_scope('Embedding'):
				self.embedded_chars = tf.nn.embedding_lookup(self.pretrained_embeddings, input_x)
		return self.embedded_chars
	@property
	def vars(self):
		return [var for var in tf.global_variables() if self.name in var.name]


class Text_Encoder(object):
	def __init__(self, args, name, vocab_size, hidden_size):
		self.args = args
		self.name = name
		self.vocab_size = vocab_size
		self.hidden_size = hidden_size
		'''
		self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
		self.input_y = tf.placeholder(tf.int64, [None, ], name="input_y")
		self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
		'''
	def __call__(self, input_x, dropout_keep_prob): # embedded input
		with tf.variable_scope(self.name):
			with tf.variable_scope('RNN_Encoder'):
				# print('len(self.hidden_size): ', len(self.hidden_size))
				if len(self.hidden_size) == 1:
					rnn_cell = tf.nn.rnn_cell.GRUCell(self.hidden_size[0])
					# 'outputs' is a tensor of shape [batch_size, max_time, cell_state_size]
					# defining initial state
					#initial_state = rnn_cell.zero_state(tf.shape(input_x)[0], dtype=tf.float32)
					# 'state' is a tensor of shape [batch_size, cell_state_size] , initial_state=initial_state
					outputs, state = tf.nn.dynamic_rnn(rnn_cell, input_x, dtype=tf.float32)
				else:
					# create N LSTMCells
					rnn_layers = [tf.nn.rnn_cell.GRUCell(size) for size in self.hidden_size]
					# create a RNN cell composed sequentially of a number of RNNCells
					multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
					# 'outputs' is a tensor of shape [batch_size, max_time, 256]
					# 'state' is a N-tuple where N is the number of LSTMCells containing a
					# tf.contrib.rnn.LSTMStateTuple for each cell
					outputs, state = tf.nn.dynamic_rnn(cell=multi_rnn_cell, inputs=input_x, dtype=tf.float32)
				encoder_state = dropout(state, dropout_keep_prob)
				# print('outputs shape :', outputs.shape)
				# print('state shape :', encoder_state.shape)
			return encoder_state

	@property
	def vars(self):
		return [var for var in tf.global_variables() if self.name in var.name]

class Style_Encoder(object):
	def __init__(self, args, name, sequence_length, embedding_size, num_filters, lambda_l2):
		self.args = args
		self.name = name
		self.sequence_length = sequence_length
		self.embedding_size = embedding_size
		self.num_filters = num_filters # 128	
		self.filter_size = [2,3,4]
		self.num_classes = 2
		self.class_weights = tf.constant([1.0, 1.0])
		self.l2_loss = tf.constant(0.0)
		self.lambda_l2 = lambda_l2

	def __call__(self, embedded_chars, input_y, dropout_keep_prob):
		with tf.variable_scope(self.name):
			with tf.variable_scope('Embedding'):
				self.embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
				# print('embedded_chars_expanded.shape : ', self.embedded_chars_expanded.shape)
			pooled_outputs = []
			for index, filter_size in enumerate(self.filter_size):
				with tf.variable_scope('Conv' + str(index)):
					W_conv = weight_variables([filter_size, self.embedding_size, 1, self.num_filters], name = 'W_conv' + str(index))
					b_conv = bias_variables([self.num_filters], name = 'b_conv' + str(index))
					h_conv = conv2d(self.embedded_chars_expanded, W_conv, 1, 'VALID') + b_conv
					conv = lrelu(h_conv) # shape : vocab_size,embedding_size,128
					# print('conv shape : ', conv.shape)
					pooled = max_pool(conv, self.sequence_length - filter_size + 1, 1, 'VALID')
					pooled = dropout(pooled, dropout_keep_prob)
					#pooled = max_pool(pooled, 1, self.num_filters, 'VALID')
					pooled_outputs.append(pooled)
					# print('pooled shape : ', pooled.shape)

			# Combine all the pooled features
			num_filters_total = self.num_filters * len(self.filter_size) # 128, 128, 128
			self.h_pool = tf.concat(pooled_outputs, 3)
			self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
			# print(self.h_pool_flat.shape)

			# Add dropout
			with tf.name_scope("dropout"):
				self.h_drop = dropout(self.h_pool_flat, dropout_keep_prob)
			with tf.name_scope("output"):
				W_output = weight_variables([num_filters_total, self.num_classes], name = 'W_output')
				b_output = bias_variables([self.num_classes], name = 'b_output')
				self.l2_loss += tf.nn.l2_loss(W_output)
				self.l2_loss += tf.nn.l2_loss(b_output)
				self.scores = tf.matmul(self.h_drop, W_output) + b_output
				self.predictions = tf.argmax(tf.nn.softmax(self.scores), 1, name="predictions")
			# Calculate mean cross-entropy loss
			with tf.name_scope("loss"):
				# print('score dim : ', self.scores.shape)
				# print('input_y dim : ', input_y.shape)
				balanced_weight = tf.gather(self.class_weights, input_y)
				losses = tf.losses.sparse_softmax_cross_entropy(input_y, self.scores, weights=balanced_weight)
				self.loss = tf.reduce_mean(losses) + self.lambda_l2*self.l2_loss
			# Accuracy
			with tf.name_scope("accuracy"):
				correct_predictions = tf.equal(self.predictions, input_y)
				self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

			return self.h_pool_flat, self.loss, self.accuracy, self.predictions
	@property
	def vars(self):
		return [var for var in tf.global_variables() if self.name in var.name]

class Text_Generator(object):
	def __init__(self, args, name, sequence_length, vocab_size, g_hidden_size):
		self.args = args
		self.name = name
		self.sequence_length = sequence_length
		self.vocab_size = vocab_size
		self.g_hidden_size = g_hidden_size
		self.proj_w = weight_variables([self.g_hidden_size, self.vocab_size], name='proj_w')
		self.proj_b = bias_variables([self.vocab_size], name='proj_b')
	def __call__(self, teacher_force = False, text_embedding, input_x, encoder_state, embedded_style, dropout_keep_prob, temperature, reuse_variables=True):
		with tf.variable_scope(self.name, reuse=reuse_variables):
			h_state, output_seq = [], []
			rnn_cell = tf.nn.rnn_cell.GRUCell(self.g_hidden_size)
			# 'outputs' is a tensor of shape [batch_size, max_time, cell_state_size]
			# defining initial state
			initial_state = tf.concat([encoder_state, embedded_style], 1)
			h_state.append(initial_state)
			if not teacher_force:
				# print('Shape of initial state: ', initial_state.shape)
				# 'state' is a tensor of shape [batch_size, cell_state_size]
				# print('Shape of input for rnn: ', self.reshape_text_embedding(text_embedding, input_x).shape)
				outputs, state = tf.nn.dynamic_rnn(rnn_cell, self.reshape_text_embedding(text_embedding, input_x), initial_state=initial_state, dtype=tf.float32)
				outputs = self.decode(outputs, dropout_keep_prob, temperature)
				output_seq.append(outputs)
				h_state.append(state)
				for i in range(1, self.sequence_length):
					tf.get_variable_scope().reuse_variables()
					outputs, state = tf.nn.dynamic_rnn(rnn_cell, self.reshape_text_embedding(text_embedding, outputs), initial_state=state, dtype=tf.float32)
					outputs = self.decode(outputs, dropout_keep_prob, temperature)
					output_seq.append(outputs)
					h_state.append(state)
				return output_seq, h_state
			else:
				outputs, state = tf.nn.dynamic_rnn(rnn_cell, self.reshape_text_embedding(text_embedding, input_x[:,0]), initial_state=initial_state, dtype=tf.float32)
				h_state.append(state)
				for i in range(1, self.sequence_length):
					tf.get_variable_scope().reuse_variables()
					outputs, state = tf.nn.dynamic_rnn(rnn_cell, self.reshape_text_embedding(text_embedding, input_x[:,i]), initial_state=state, dtype=tf.float32)
					h_state.append(state)
				return h_state

	def decode(self, outputs, dropout_keep_prob, temperature):
		outputs = dropout(tf.reshape(outputs, [-1, tf.shape(outputs)[-1]]), dropout_keep_prob)
		outputs = tf.matmul(outputs, self.proj_w) + self.proj_b
		outputs = gumbel_softmax(outputs, temperature)
		return outputs
	def reshape_text_embedding(self, text_embedding, input_x):
		return tf.expand_dims(text_embedding(input_x), 1)
	@property
	def vars(self):
		return [var for var in tf.global_variables() if self.name in var.name]



class Discriminator(object):
	def __init__(self, args, name, sequence_length, embedding_size, num_filters):
		self.args = args
		self.name = name
		self.sequence_length = sequence_length
		self.embedding_size = embedding_size
		self.num_filters = num_filters # 128	
		self.filter_size = [2,3,4]
		self.num_classes = 2
		self.class_weights = tf.constant([1.0, 1.0])

	def __call__(self, hidden_state, dropout_keep_prob, reuse_variables=True):
		with tf.variable_scope(self.name, reuse=reuse_variables):
		
			self.embedded_chars_expanded = tf.expand_dims(hidden_state, -1)
				# print('embedded_chars_expanded.shape : ', self.embedded_chars_expanded.shape)
			pooled_outputs = []
			for index, filter_size in enumerate(self.filter_size):
				with tf.variable_scope('Conv' + str(index)):
					W_conv = weight_variables([filter_size, self.embedding_size, 1, self.num_filters], name = 'W_conv' + str(index))
					b_conv = bias_variables([self.num_filters], name = 'b_conv' + str(index))
					h_conv = conv2d(self.embedded_chars_expanded, W_conv, 1, 'VALID') + b_conv
					conv = lrelu(h_conv) # shape : vocab_size,embedding_size,128
					# print('conv shape : ', conv.shape)
					pooled = max_pool(conv, self.sequence_length - filter_size + 1, 1, 'VALID')
					pooled = dropout(pooled, dropout_keep_prob)
					#pooled = max_pool(pooled, 1, self.num_filters, 'VALID')
					pooled_outputs.append(pooled)
					# print('pooled shape : ', pooled.shape)

			# Combine all the pooled features
			num_filters_total = self.num_filters * len(self.filter_size) # 128, 128, 128
			self.h_pool = tf.concat(pooled_outputs, 3)
			self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
			# print(self.h_pool_flat.shape)

			# Add dropout
			self.h_drop = dropout(self.h_pool_flat, dropout_keep_prob)

			W_output = weight_variables([num_filters_total, self.num_classes], name = 'W_output')
			b_output = bias_variables([self.num_classes], name = 'b_output')
			self.scores = tf.matmul(self.h_drop, W_output) + b_output

			return self.scores
	@property
	def vars(self):
		return [var for var in tf.global_variables() if self.name in var.name]
