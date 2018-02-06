import tensorflow as tf
import numpy as np
from network import TextCNN
import argparse,sys
import time
from tensorflow.contrib import learn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class Batch_generation(object):
	def __init__(self, x_train, y_train, x_test, y_test):
		self.x_train = x_train
		self.y_train = y_train
		self.x_test = x_test
		self.y_test = y_test
		self.amount = int(len(self.x_train))
		self._index = 0
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
	training_loss_list = []
	valid_loss_list = []
	training_acc_list = []
	valid_acc_list = []
	plt.ion()
	if args.train:
		sess = tf.Session()
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver()
		writer = tf.summary.FileWriter("logs/", sess.graph)
		for ep in range(batch_epoch):
			timestamp1 = time.time()
			x_train_batch, y_train_batch = batch_generation(batch_size, True)
			feed_dict = {textcnn.input_x: x_train_batch, textcnn.input_y: y_train_batch, textcnn.dropout_keep_prob: 0.15}
			_, loss, accuracy = sess.run([textcnn.train_op, textcnn.loss, textcnn.accuracy], feed_dict=feed_dict)
			training_loss_list.append(loss)
			training_acc_list.append(accuracy)
			timestamp2 = time.time()
			print('Epoch : ', ep, '\ttime %.2f: ' % (timestamp2 - timestamp1), '\tLOSS : ', loss, '\tACC : ', accuracy)
			if (ep+1) % 100 == 0:
				# test validation			
				timestamp1 = time.time()
				x_test, y_test = batch_generation(batch_size, False)
				feed_dict = {textcnn.input_x: x_test, textcnn.input_y: y_test, textcnn.dropout_keep_prob: 1.0}
				loss, accuracy = sess.run([textcnn.loss, textcnn.accuracy], feed_dict=feed_dict)
				valid_loss_list.append(loss)
				valid_acc_list.append(accuracy)
				
				timestamp2 = time.time()
				print("Validddddddd!!!!!")
				print("==============")
				print('Epoch : ', ep, '\ttime %.2f: ' % (timestamp2 - timestamp1), '\tLOSS : ', loss, '\tACC : ', accuracy)
				print("==============")
				print("SAVEEEEE MODELLLLLLL")
				saver.save(sess, "Model/model.ckpt")
			if (ep+1) % 5000 == 0:
				plt.figure(1)
				#plt.title('training')
				plt.subplot(211)
				plt.plot(np.arange(len(training_loss_list)),training_loss_list)
				plt.subplot(212)
				plt.plot(np.arange(len(training_acc_list)),training_acc_list)
				plt.pause(1e-18)
				plt.show()
				plt.figure(2)
				#plt.title('valid')
				plt.subplot(211)
				plt.plot(np.arange(len(valid_loss_list)),valid_loss_list)
				plt.subplot(212)
				plt.plot(np.arange(len(valid_acc_list)),valid_acc_list)
				plt.pause(1e-18)
				plt.show()
if __name__ == '__main__':
	parser = argparse.ArgumentParser('')
	parser.add_argument('--train', action='store_true', help='whether train TextCNN')
	parser.add_argument('--test', action='store_true', help='whether test TextCNN')
	args = parser.parse_args()

	data_dir = '/home/zong-ze/Research/Emotion_classification/dataset/crowdflower/train_data/'
	# X_train = np.load(data_dir+'x_train.npy')
	# y_train = np.load(data_dir+'y_train.npy')
	# X_test = np.load(data_dir+'X_test.npy')
	# y_test = np.load(data_dir+'y_test.npy')
	data = np.load(data_dir+'data_origin.npy')
	label = np.load(data_dir+'label.npy')
	print('data amount : ', len(data))
	print('label amount : ', len(label))
	print(data[0:5])
	vocab_processor = learn.preprocessing.VocabularyProcessor(21,min_frequency=4)
	x = np.array(list(vocab_processor.fit_transform(data)))
	print(x[0:5])
	print('vocab_size : ', len(vocab_processor.vocabulary_))
	X_train, X_test, y_train, y_test = train_test_split(x, label, test_size=0.2, random_state=42)
	batch_size = 128
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
	print('X_train : ', len(y_train))
	print('X_test : ', len(y_test))
	time.sleep(10)
	batch_generation =  Batch_generation(X_train, y_train, X_test, y_test)
	print('Build batch generation')
	textcnn = TextCNN(sequence_length, num_classes, vocab_size, embedding_size, num_filters, lambda_l2)
	main(args, textcnn, batch_generation, batch_size)		
