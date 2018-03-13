import tensorflow as tf
import numpy as np
from network import TextCNN
import argparse,sys
import time, random
from tensorflow.contrib import learn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import gensim
from ops import *

class Batch_generation(object):
	def __init__(self, x_train, y_train, x_test, y_test):
		self.x_train = x_train
		self.y_train = y_train
		self.x_test = x_test
		self.y_test = y_test
		self.amount = int(len(self.x_train))
		self.test_amount = int(len(self.y_test)) - 1
		self._index = 0
	def __call__(self, batch_size, train):
		if train:
			self.index = self._index%self.amount
			x_train_batch = self.x_train[self.index:self.index+batch_size]
			y_train_batch = self.y_train[self.index:self.index+batch_size]
			self._index += batch_size 

			return x_train_batch, y_train_batch
		else:
			rand_index = random.sample(range(self.test_amount), batch_size)
			#return self.x_test[rand_index], self.y_test[rand_index]
			return [self.x_test[i] for i in rand_index], [self.y_test[i] for i in rand_index]

def main(args, sess, textcnn, batch_generation, batch_size, batch_epoch = 1000000):
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
				loss, accuracy, predictions = sess.run([textcnn.loss, textcnn.accuracy, textcnn.predictions], feed_dict=feed_dict)
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
				fig1 = plt.figure(1)
				#plt.title('training')
				plt.subplot(211)
				plt.plot(np.arange(len(training_loss_list)),training_loss_list, 'b')
				y = [np.mean(training_loss_list[:i]) if i < 30 else np.mean(training_loss_list[(i-29):(i+1)]) for i in range(1, len(training_loss_list)+1) ]
				plt.plot(np.arange(len(training_loss_list)),y,'r')
				plt.subplot(212)
				plt.plot(np.arange(len(training_acc_list)),training_acc_list, 'b')
				y1 = [np.mean(training_acc_list[:i]) if i < 30 else np.mean(training_acc_list[(i-29):(i+1)]) for i in range(1, len(training_acc_list)+1) ]
				plt.plot(np.arange(len(training_acc_list)),y1,'r')
				plt.pause(1e-18)
				plt.show()
				fig1.savefig('Train'+'.png')
				fig2 = plt.figure(2)
				#plt.title('valid')
				plt.subplot(211)
				plt.plot(np.arange(len(valid_loss_list)),valid_loss_list, 'b')
				y = [np.mean(valid_loss_list[:i]) if i < 30 else np.mean(valid_loss_list[(i-29):(i+1)]) for i in range(1, len(valid_loss_list)+1) ]
				plt.plot(np.arange(len(valid_loss_list)),y,'r')
				plt.subplot(212)
				plt.plot(np.arange(len(valid_acc_list)),valid_acc_list, 'b')
				y1 = [np.mean(valid_acc_list[:i]) if i < 30 else np.mean(valid_acc_list[(i-29):(i+1)]) for i in range(1, len(valid_acc_list)+1) ]
				plt.plot(np.arange(len(valid_acc_list)),y1,'r')
				plt.pause(1e-18)
				plt.show()
				fig2.savefig('Valid'+'.png')

				cm = tf.contrib.metrics.confusion_matrix(y_test, predictions, num_classes = 2).eval()
				fig3 = plt.figure()
				class_names = ['positive', 'negative']
				plot_confusion_matrix(cm, classes=class_names, normalize=True,
					title='Normalized confusion matrix')
				plt.pause(1e-18)
				plt.show()
				fig3.savefig('confusion_matrix'+'.png')
if __name__ == '__main__':
	parser = argparse.ArgumentParser('')
	parser.add_argument('--train', action='store_true', help='whether train TextCNN')
	parser.add_argument('--test', action='store_true', help='whether test TextCNN')
	args = parser.parse_args()

	data_dir = './../'
	pos_data = []
	neg_data = []
	# with open(data_dir+'positive_sentence.csv') as myfile:
	# 	for line in myfile:
	# 		pos_data.append(line.strip())
	# with open(data_dir+'negative_sentence.csv') as myfile:
	# 	for line in myfile:
	# 		neg_data.append(line.strip())
	# for dataset from others' github
	with open(data_dir+'../LST_examples/language-style-transfer/data/yelp/sentiment.train.1') as myfile:
		for line in myfile:
			pos_data.append(line.strip())
	with open(data_dir+'../LST_examples/language-style-transfer/data/yelp/sentiment.train.0') as myfile:
		for line in myfile:
			neg_data.append(line.strip())

	# rand_index = random.sample(range(len(pos_data)-1), len(neg_data))
	# pos_data_1 = [pos_data[i] for i in rand_index]
	# pos_data = pos_data_1
	print('len of pos_data :', len(pos_data))
	print('len of neg_data :', len(neg_data))
	data = pos_data + neg_data
	label = [0]*len(pos_data)+[1]*len(neg_data)
	print('data amount : ', len(data))
	print('label amount : ', len(label))
	print(data[0:5])

	# load pretrain genism model
	timestamp1 = time.time()
	model = gensim.models.KeyedVectors.load_word2vec_format(data_dir+'../Emotion_classification/src/gensim_word2vec/model/GoogleNews-vectors-negative300.bin', binary=True) 
	timestamp2 = time.time()
	print('Time for load model %.2f: ' % (timestamp2 - timestamp1))


	vocab_processor = learn.preprocessing.VocabularyProcessor(15,min_frequency=5)
	x = np.array(list(vocab_processor.fit_transform(data)))
	print(x[0:5])
	print('vocab_size : ', len(vocab_processor.vocabulary_))
	my_embedding_matrix = np.zeros(shape=(len(vocab_processor.vocabulary_), 300)) # embedding_size = 512

	for word in vocab_processor.vocabulary_._mapping:
		id = vocab_processor.vocabulary_._mapping[word]
		if word in model.wv.vocab:
			my_embedding_matrix[id] = model.wv[word]
		else:
			my_embedding_matrix[id] = np.random.uniform(low=-0.25, high=0.25, size=300)
	X_train, X_test, y_train, y_test = train_test_split(x, label, test_size=0.2, random_state=42)
	batch_size = 64
	if args.train:
		sequence_length = len(X_train[0])
		num_classes = 2
		vocab_size = len(vocab_processor.vocabulary_)
		embedding_size = 300
		num_filters = 256
		dropout_keep_prob = 0.8
		lambda_l2 = 0.01
	if args.test:
		sequence_length = len(X_train[0])
		num_classes = 2
		vocab_size = len(vocab_processor.vocabulary_)
		embedding_size = 300
		num_filters = 256
		dropout_keep_prob = 1
		lambda_l2 = 0.01
	print('X_train : ', len(y_train))
	print('X_test : ', len(y_test))
	time.sleep(10)
	batch_generation =  Batch_generation(X_train, y_train, X_test, y_test)
	print('Build batch generation')
	textcnn = TextCNN(sequence_length, num_classes, vocab_size, embedding_size, num_filters, lambda_l2)

	pretrained_embeddings = tf.placeholder(tf.float32, [None, None], name="pretrained_embeddings")
	set_x = textcnn.pretrained_embeddings.assign(pretrained_embeddings)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		sess.run(set_x, feed_dict={pretrained_embeddings: my_embedding_matrix})
		main(args, sess, textcnn, batch_generation, batch_size)		
	
#'./../../../dataset/SemEval-2018/El-reg-En-train/train_data/'