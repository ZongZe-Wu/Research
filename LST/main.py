from data_preprocess import Build_Vocab
from RNN_Network import Text_Embedding, Text_Encoder, Style_Encoder, Text_Generator, Discriminator
import argparse
import numpy as np
import random
import tensorflow as tf




class Style_transfer(object):
	def __init__(self, args, sess, vocab_build, epoch, text_embedding, text_encoder, style_encoder, text_generator, d1, d2):
		self.args = args
		self.sess = sess
		self.epoch = epoch
		self.vocab_build = vocab_build
		self.text_embedding = text_embedding
		self.text_encoder = text_encoder
		self.style_encoder = style_encoder
		self.text_generator = text_generator
		self.d1 = d1
		self.d2 = d2

		self.dropout_keep_prob = 0.8
		self.temperature = tf.placeholder(tf.float32, name="temperature")
		self.input_x = tf.placeholder(tf.int32, [None, self.vocab_build.max_len_x+1], name="input_x")
		self.style_label = tf.placeholder(tf.int64, [None, ], name="style_label")


		self.embedded_chars = self.text_embedding(self.input_x)
		print('=====Shape of text embedding output: ', self.embedded_chars.shape)

		self.encoded_state = self.text_encoder(self.embedded_chars, self.dropout_keep_prob)
		print('=====Shape of text encoder output: ', self.encoded_state.shape)

		self.embedded_style, self.style_loss, self.style_accuracy, self.style_predictions = \
		self.style_encoder(self.embedded_chars, self.style_label, self.dropout_keep_prob) # embedded_chars, input_y, dropout_keep_prob
		print('=====Shape of style encoder output: ', self.embedded_style.shape)
		#                 <bos>
		# text_embedding, input_x, encoder_state, embedded_style, dropout_keep_prob, temperature
		self.gen_seq, self.gen_h_state = \
		self.text_generator(self.text_embedding, self.input_x[:,0], self.encoded_state, self.embedded_style, self.dropout_keep_prob, self.temperature, reuse_variables=False)
		print('=====Shape of generated sequence: ', np.array(self.gen_seq).shape)
		print('=====Shape of generated hidden state : ', np.array(self.gen_h_state).shape)

		self.tf_gen_seq, self.tf_gen_h_state = \
		self.text_generator(teacher_force=True, self.text_embedding, self.input_x, self.encoded_state, self.embedded_style, self.dropout_keep_prob, self.temperature)

		# distinguish between real x1 and transfered x2
		self.d_1 = self.d1(self.tf_gen_seq, self.dropout_keep_prob, reuse_variables=False)
		self.d_1_transferred = self.d1(self.gen_seq, self.dropout_keep_prob)

		# distinguish between real x2 and transfered x1
		self.d_1 = self.d1(self.tf_gen_seq, self.dropout_keep_prob, reuse_variables=False)
		self.d_1_transferred = self.d1(self.gen_seq, self.dropout_keep_prob)


		if args.train:
			pretrained_embeddings = tf.placeholder(tf.float32, [None, None], name="pretrained_embeddings")
			set_x = self.text_embedding.pretrained_embeddings.assign(pretrained_embeddings)
			self.sess.run(set_x, feed_dict={pretrained_embeddings: self.vocab_build.my_embedding_matrix})			
			writer = tf.summary.FileWriter("logs/", self.sess.graph)
			saver = tf.train.Saver()
			# self.train()

	def train(self):
		for ep in range(self.epoch):
			iteration = 0
			for input_sent, output_sent, label in self.vocab_build.batch_generation():
				print('haha')	




















if __name__ == '__main__':
	parser = argparse.ArgumentParser('')
	parser.add_argument('--train', action='store_true', help='whether train')
	parser.add_argument('--test', action='store_true', help='whether test')
	parser.add_argument('--vocab', type=str, default='', help='whether exist vocab.npz')
	parser.add_argument('--model', type=str, default='', help='whether there is a model')
	args = parser.parse_args()

	if args.train:
		batch_size = 32
		epoch = 50
		max_len_x = 15
		text_encoder_hidden_size = [256]

	data = ['sentiment.train.1', 'sentiment.train.0']
	# def __init__(self, args, embed_model, data, batch_size, max_len_x, min_occur=5)
	vocab_build = Build_Vocab(args, True, data, batch_size, max_len_x)

	# def __init__(self, args, name, vocab_size, embedding_size):
	text_embedding = Text_Embedding(args, 'Text_Embedding', vocab_build.vocab_size, vocab_build.embedding_size) 
	print('===Build Text Embedding Model===')
	# def __init__(self, args, name, vocab_size, hidden_size):
	text_encoder = Text_Encoder(args, 'Text_Encoder', vocab_build.vocab_size, text_encoder_hidden_size)
	print('===Build Text Encoder Model===')
	# def __init__(self, args, name, sequence_length, embedding_size, num_filters, lambda_l2)
	style_encoder = Style_Encoder(args, 'Style_Encoder', max_len_x + 1, vocab_build.embedding_size, 128, 0.01)
	print('===Build Style Encoder Model===')
	# def __init__(self, args, name, sequence_length, vocab_size, g_hidden_size):
	text_generator = Text_Generator(args, 'Text_Generator', max_len_x + 1, vocab_build.vocab_size, 640) # concat of text encoder output shape : text_encoder_hidden_size(256) and style encoder output shape: num_filters*3 (128*3)
	print('===Build Text Generator Model===')
	# def __init__(self, args, name, sequence_length, embedding_size, num_filters)
	d1 = Discriminator(args, 'D1',  max_len_x + 1 + 1, 640, 128)
	d2 = Discriminator(args, 'D2',  max_len_x + 1 + 1, 640, 128)
	with tf.Session() as sess:
		# tf.reset_default_graph()
		sess.run(tf.global_variables_initializer())	
		style_transfer = Style_transfer(args, sess, vocab_build, epoch, text_embedding, text_encoder, style_encoder, text_generator, d1, d2)






