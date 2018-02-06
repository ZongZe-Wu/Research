import tensorflow as tf
import numpy as np
from network import TextCNN
import argparse,sys
import time


class Batch_generation(object):
	def __init__(self, x_train, y_train, x_test, y_test):
		self.x_train = x_train
		self.y_train = y_train
		self.x_test = x_test
		self.y_test = y_test
		self.amount = int(len(self.x_train))
		self.index = 0
	def __call__(self, batch_size, train):
		if train:
			self.index = self._index%self.amount
			x_train_batch = self.x_train[self.index:self.index+batch_size]
			y_train_batch = self.y_train[self.index:self.index+batch_size]
			self._index += batch_size 

			return x_train_batch, y_train_batch
		else:
			return self.x_test, self.y_test

def main(args, textcnn, batch_generation, batch_size, batch_epoch = 100000):
	# sequence_length = len(X_train[0])
	# num_classes = 13
	# vocab_size = 5567
	# embedding_size = 256
	# num_filters = 128
	# dropout_keep_prob = 0.8
	# lambda_l2 = 0.01
	x = tf.placeholder(tf.int32, [None, 21], name='x')
	y = tf.placeholder(tf.float32, [None, 13], name="y")
	if args.train:
		sess = tf.Session()
		writer = tf.summary.FileWriter("logs/", self.sess.graph)
		for ep in range(batch_epoch):
			timestamp1 = time.time()
			x_train_batch, y_train_batch = batch_generation(batch_size, True)
			feed_dict = {x: x_train_batch, y: y_train_batch}
			_, loss, accuracy = sess.run([self.train_op, self.loss, self.accuracy], feed_dict=feed_dict)
			timestamp2 = time.time()
			print('Epoch : ', ep, '\ttime %.2f: ' % (timestamp2 - timestamp1), '\tLOSS : ', loss, '\tACC : ', accuracy)
			if (ep+1) % 100 == 0:
				# test validation
				timestamp1 = time.time()
				x_test, y_test = batch_generation(batch_size, False)
				feed_dict = {x: x_test, y: y_test}
				loss, accuracy = sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
				timestamp2 = time.time()
				print('Epoch : ', ep, '\ttime %.2f: ' % (timestamp2 - timestamp1), '\tLOSS : ', loss, '\tACC : ', accuracy)
				print("SAVEEEEE MODELLLLLLL")
				saver.save(sess, "Model/model.ckpt")

if __name__ == '__main__':
	parser = argparse.ArgumentParser('')
	parser.add_argument('--train', action='store_true', help='whether train TextCNN')
	parser.add_argument('--test', action='store_true', help='whether test TextCNN')
	args = parser.parse_args()

	data_dir = '/home/zong-ze/Research/Emotion_classification/dataset/crowdflower/train_data/'
	X_train = np.load(data_dir+'x_train.npy')
	y_train = np.load(data_dir+'y_train.npy')
	X_test = np.load(data_dir+'x_test.npy')
	y_test = np.load(data_dir+'y_test.npy')
	batch_size = 32
	if args.train:
		sequence_length = len(X_train[0])
		num_classes = 13
		vocab_size = 5567
		embedding_size = 256
		num_filters = 128
		dropout_keep_prob = 0.8
		lambda_l2 = 0.01
	if args.test:
		sequence_length = len(X_train[0])
		num_classes = 13
		vocab_size = 5567
		embedding_size = 256
		num_filters = 128
		dropout_keep_prob = 1
		lambda_l2 = 0.01
	print('X_train amount : ', len(X_train))
	print('X_test amount : ', len(X_test))
	batch_generation =  Batch_generation(X_train, y_train, X_test, y_test )
	print('Build batch generation')
	textcnn = TextCNN(sequence_length, num_classes, vocab_size, embedding_size, num_filters, dropout_keep_prob, lambda_l2)
	main(args, textcnn, batch_generation, batch_size)		
