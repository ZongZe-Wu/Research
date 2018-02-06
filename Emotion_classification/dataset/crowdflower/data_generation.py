import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import itertools
from token_test import TreebankWordTokenizer as TB
from scipy.sparse import csr_matrix, find
from sklearn.model_selection import train_test_split

MAX_LEN_SEQUENCE = 21
'''
def batch_generation(data, batch_size):
	for i in range(len(data)):
		sentence_buf = []
		#it = 0
		for j in range(len(data[i])):
			#print(data[i])
			#print(TB().tokenize(data[i][j]))
			buf = TB().tokenize(data[i][j])
			print(buf)
			for k in range(len(buf)):
				#it += 1
				if buf[k].lower() in vectorizer.stop_words_:
					buf[k] = "unk"
				else:
					buf[k] = buf[k].lower()
				#print(buf[k])
				one_hot_buf = [0]*max_voc_size
				#print((find(vectorizer.transform([buf[k]]))[1]))
				one_hot_buf[int(find(vectorizer.transform([buf[k]]))[1])] = 1
				#print(find(vectorizer.transform([buf[k]]))[1])
				sentence_buf.append(one_hot_buf)
		#print(it)
		
		print(len(sentence_buf))
		print(sentence_buf[0].index(1))
		print(sentence_buf[1].index(1))	
		print(sentence_buf[2].index(1))
		
		# padddddddddding
		if len(sentence_buf) < MAX_LEN_SEQUENCE:
			pad_zero = [[0]*max_voc_size]*(MAX_LEN_SEQUENCE-len(sentence_buf))
			sentence_buf = sentence_buf + pad_zero
		if len(sentence_buf) != MAX_LEN_SEQUENCE:
			#print(pad_zero)
			print(len(sentence_buf))
			print(data[i])
			print(len(data[i]))
			print('errrrrrrr!!!')
			break
		print(len(sentence_buf))
		training_data.append(sentence_buf)	
		#print(training_data)
	print(len(training_data))
'''
def find_max_length(a, b):
	if a >= b:
		return a
	else:
		return b
def main():
	print('MAX_LEN_SEQUENCE : ', MAX_LEN_SEQUENCE)
	data_path = '/home/zong-ze/Research/Emotion_classification/dataset/crowdflower/preprocess_data/'
	emotion_filename = ['anger', 'boredom', 'empty', 'enthusiasm', 'fun', 'happiness', 'hate', 'love', 'neutral', 'relief', 'sadness', 'surprise', 'worry']
	data = []
	data_origin = []

	label = []
	max_len = 0
	for emo in emotion_filename:
		with open(data_path+emo+'.csv') as myfile:
			for line in myfile:
				buf = line.strip().split(' ')
				buf = list(filter(str.strip, buf))
				max_len = find_max_length(max_len, len(buf))
				if len(buf) <= MAX_LEN_SEQUENCE:
					buf1 = ''
					for i in range(len(buf)):
						buf1 += buf[i]+' '
					print(buf1)
					data_origin.append(buf1)
					data.append(buf)
					label.append(emotion_filename.index(emo))
	print('data length : ', len(data)) # 40000
	print('label length : ', len(label)) # 40000
	print('max len : ', max_len)
	print(data[1])
	np.save('train_data/data_origin.npy', data_origin)
	np.save('train_data/label.npy', label)
	X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.1, random_state=42)
	# np.save('train_data/x_train.npy', X_train)
	# np.save('train_data/y_train.npy', y_train)
	# np.save('train_data/X_test.npy', X_test)
	# np.save('train_data/y_test.npy', y_test)
	# Countvectorizer

	vectorizer = CountVectorizer(tokenizer=TB().tokenize, min_df = 4)
	vectorizer.fit(list(itertools.chain.from_iterable(data)))
	#print(vectorizer.transform(["Fridaaaayyyyy"]))
	#print(find(vectorizer.transform(["Fridaaaayyyyy"]))[1])
	#inverse = [(value, key) for key, value in vectorizer.vocabulary_.items()]
	max_voc_size = len(vectorizer.vocabulary_)
	vectorizer.vocabulary_["unk"] = max_voc_size
	vectorizer.vocabulary_["eos"] = max_voc_size + 1
	max_voc_size =  len(vectorizer.vocabulary_)
	print(max_voc_size)
	inv_map = {v: k for k, v in vectorizer.vocabulary_.items()}
	np.save('train_data/voc_dict.npy', vectorizer.vocabulary_)
	np.save('train_data/inv_voc_dict.npy', inv_map)
	training_data = []

	for i in range(len(data)):
		sentence_buf = []
		#it = 0
		for j in range(len(data[i])):
			#print(data[i])
			#print(TB().tokenize(data[i][j]))
			buf = TB().tokenize(data[i][j])
			#print(buf)
			for k in range(len(buf)):
				#it += 1
				if buf[k].lower() in vectorizer.stop_words_:
					buf[k] = "unk"
				else:
					buf[k] = buf[k].lower()
				#print(buf[k])
				sentence_buf.append(buf[k])
		#print(it)				
		#padddddddddding
		if len(sentence_buf) < MAX_LEN_SEQUENCE:
			pad_zero = ["eos"]*(MAX_LEN_SEQUENCE-len(sentence_buf))
			sentence_buf = sentence_buf + pad_zero
		if len(sentence_buf) != MAX_LEN_SEQUENCE:
			#print(pad_zero)
			print(len(sentence_buf))
			print(data[i])
			print(len(data[i]))
			print('errrrrrrr!!!')
			break
		#print(len(sentence_buf))
		#print(sentence_buf)
		training_data.append(sentence_buf)	
		#print(training_data)
	print(len(training_data))
	X_train, X_test, y_train, y_test = train_test_split(training_data, label, test_size=0.1, random_state=42)
	np.save('train_data/x_train.npy', X_train)
	np.save('train_data/y_train.npy', y_train)
	np.save('train_data/X_test.npy', X_test)
	np.save('train_data/y_test.npy', y_test)

	#if 'fridaaaayyyyy' in vectorizer.stop_words_:
	#	print('lichen fuck')
	#print(vectorizer.stop_words_)
	'''
	for i in range(len(data)):
		sentence_buf = []
		#it = 0
		for j in range(len(data[i])):
			#print(data[i])
			#print(TB().tokenize(data[i][j]))
			buf = TB().tokenize(data[i][j])
			print(buf)
			for k in range(len(buf)):
				#it += 1
				if buf[k].lower() in vectorizer.stop_words_:
					buf[k] = "unk"
				else:
					buf[k] = buf[k].lower()
				#print(buf[k])
				one_hot_buf = [0]*max_voc_size
				#print((find(vectorizer.transform([buf[k]]))[1]))
				one_hot_buf[int(find(vectorizer.transform([buf[k]]))[1])] = 1
				#print(find(vectorizer.transform([buf[k]]))[1])
				sentence_buf.append(one_hot_buf)
		#print(it)
		
		print(len(sentence_buf))
		print(sentence_buf[0].index(1))
		print(sentence_buf[1].index(1))	
		print(sentence_buf[2].index(1))
		
		# padddddddddding
		if len(sentence_buf) < MAX_LEN_SEQUENCE:
			pad_zero = [[0]*max_voc_size]*(MAX_LEN_SEQUENCE-len(sentence_buf))
			sentence_buf = sentence_buf + pad_zero
		if len(sentence_buf) != MAX_LEN_SEQUENCE:
			#print(pad_zero)
			print(len(sentence_buf))
			print(data[i])
			print(len(data[i]))
			print('errrrrrrr!!!')
			break
		print(len(sentence_buf))
		training_data.append(sentence_buf)	
		#print(training_data)
	print(len(training_data))
	'''
if __name__ == '__main__':
	main()		



