# Reference :
# https://github.com/shentianxiao/language-style-transfer/blob/8eab5fa84c1eab30cd3d57ec71b97471aa5e0c17/code/vocab.py


import numpy as np
import random
from collections import Counter
from sklearn.model_selection import train_test_split
import argparse, glob, time
import gensim

class Build_Vocab(object):
	def __init__(self, args, embed_model, data, batch_size, max_len_x, min_occur=5):
		data_dir = './../LST_examples/language-style-transfer/data/yelp/'
		self.pos_data = []
		self.neg_data = []
		with open(data_dir+data[0]) as myfile:
			for line in myfile:
				self.pos_data.append(line.strip())
		with open(data_dir+data[1]) as myfile:
			for line in myfile:
				self.neg_data.append(line.strip())
		# self.data = pos_data + neg_data
		self.pos_len = len(pos_data)
		self.neg_len = len(neg_data)
		self.pos_label = [0]*self.pos_len
		self.neg_label = [1]*self.neg_len

		self.pos_x_train, self.pos_x_test, self.pos_y_train, self.pos_y_test = train_test_split(self.pos_data, self.pos_label, test_size=0.2, random_state=42)
		self.neg_x_train, self.neg_x_test, self.neg_y_train, self.neg_y_test = train_test_split(self.neg_data, self.neg_label, test_size=0.2, random_state=42)

		self.batch_size = batch_size
		self.min_occur = min_occur
		self.max_len_x = max_len_x
		print('max_len_x :', self.max_len_x)
		print('pos_len : ', self.pos_len)
		print('neg_len : ', self.neg_len)
		print('len of X_train : ', len(self.pos_y_train+self.neg_y_train))
		print('len of X_test : ', len(self.pos_y_test+self.neg_y_test))

		print(glob.glob(args.vocab))
		if len(glob.glob(args.vocab)) == 0:
			print('building vocabulary!!')
			self.build_vocabulary()
			print('finishing vocabulary building !!!')
		else:
			print('loading the vocab file')
			npzfile= np.load(args.vocab)
			self.vocab_size = npzfile['vocab_size']
			self.word2id = npzfile['word2id'][()]
			self.id2word = npzfile['id2word']
			print('type of word2id : ', type(self.word2id))
			print('finising loading')

		if embed_model:
			# load pretrain genism model
			print('loading embedding model')
			timestamp1 = time.time()
			model = gensim.models.KeyedVectors.load_word2vec_format('../Emotion_classification/src/gensim_word2vec/model/GoogleNews-vectors-negative300.bin', binary=True) 
			timestamp2 = time.time()
			print('Time for load model %.2f: ' % (timestamp2 - timestamp1))
			self.embedding_size = 300
			self.my_embedding_matrix = np.zeros(shape=(self.vocab_size, self.embedding_size)) # embedding_size
			print('building embedding matrix')
			for word in self.id2word:
				id = self.word2id[word]
				if word in model.wv.vocab:
					self.my_embedding_matrix[id] = model.wv[word]
				else:
					self.my_embedding_matrix[id] = np.random.uniform(low=-0.25, high=0.25, size=self.embedding_size)
			print('finishing embedding matrix')

	def sent2catergoical_vector(self, sentences):
		pad = self.word2id['<pad>']
		bos = self.word2id['<bos>']
		eos = self.word2id['<eos>']
		unk = self.word2id['<unk>']
		# sentences : list of sentence
		x = [x.split(' ') for x in sentences]
		x = sorted(x, key=lambda i: len(i)) # small to large

		# max_len_x = len(x[-1])

		bos_y = []
		y_eos = []
		for i in range(len(x)):
			sentence_id = [self.word2id[w] if w in self.word2id else unk for w in x[i]]
			len_x = len(x[i])

			padding = [pad]*(self.max_len_x - len_x)
			bos_x = [bos] + sentence_id + padding
			x_eos = sentence_id + [eos] + padding

			bos_y.append(bos_x)
			y_eos.append(x_eos)

		return bos_y, y_eos

	# def batch_generation(self):
	# 	for i in range(0, len(self.X_train), self.batch_size):
	# 		train_input_sent, train_output_sent = self.sent2catergoical_vector(self.X_train[i:i+self.batch_size])

	# 		yield train_input_sent, train_output_sent, self.y_train[i:i+self.batch_size]

	def batch_generation(self):

	def pos_batch(self):
		for i in range(0, len(self.pos_x_train), self.batch_size):
			train_input_sent, train_output_sent = self.sent2catergoical_vector(self.pos_x_train[i:i+self.batch_size])
			yield train_input_sent, train_output_sent, self.pos_y_train[i:i+self.batch_size]

	def neg_batch(self):
		for i in range(0, len(self.neg_x_train), self.batch_size):
			train_input_sent, train_output_sent = self.sent2catergoical_vector(self.neg_x_train[i:i+self.batch_size])
			yield train_input_sent, train_output_sent, self.neg_y_train[i:i+self.batch_size]

	def valid_batch_generation(self):
		rand_index = random.sample(range(len(self.y_test)), self.batch_size)
		valid_input_sent, valid_output_sent = self.sent2catergoical_vector([self.X_test[i] for i in rand_index])
		return valid_input_sent, valid_output_sent, [self.y_test[i] for i in rand_index]

	def build_vocabulary(self):
		self.word2id = {'<pad>': 0, '<bos>': 1, '<eos>': 2, '<unk>': 3}
		self.id2word = ['<pad>', '<bos>', '<eos>', '<unk>']
		words = [word for sentence in self.data for word in sentence.split(' ')]
		counter = Counter(words)
		for word in counter:
			if counter[word] >= self.min_occur:
				self.word2id[word] = len(self.word2id)
				self.id2word.append(word)
		self.vocab_size = len(self.word2id)
		print(self.vocab_size)
		np.savez('vocab.npz', vocab_size=self.vocab_size, word2id=self.word2id, id2word=self.id2word)

if __name__ == '__main__':
	parser = argparse.ArgumentParser('')
	parser.add_argument('--train', type=str, default='', help='whether train')
	parser.add_argument('--test', type=str, default='', help='whether test')
	parser.add_argument('--vocab', type=str, default='', help='whether exist vocab.npz')
	args = parser.parse_args()

	# data = ['positive_sentence.csv', 'negative_sentence.csv']
	data = ['sentiment.train.1', 'sentiment.train.0']
	# def __init__(self, args, embed_model, data, batch_size, max_len_x, min_occur=5)
	vocab_build = Build_Vocab(args, True, data, 32, 15)
	it = 0
	for input_sent, output_sent, label in vocab_build.batch_generation():
		print(vocab_build.X_train[0:5])
		print(input_sent)
		break
