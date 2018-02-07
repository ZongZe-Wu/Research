import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import itertools
from scipy.sparse import csr_matrix, find
from sklearn.model_selection import train_test_split

MAX_LEN_SEQUENCE = 30

def find_max_length(a, b):
	if a >= b:
		return a
	else:
		return b
def main():
	print('MAX_LEN_SEQUENCE : ', MAX_LEN_SEQUENCE)
	data_path = './train/'
	data_origin = []

	label = []
	max_len = 0
	myfile = np.load(data_path+'training_data'+'.npy')
	print(np.array(myfile).shape)
	for line in myfile:
		buf = line[1].strip().split(' ')
		text_buf = list(filter(str.strip, buf))
		max_len = find_max_length(max_len, len(text_buf))
		if len(text_buf) <= MAX_LEN_SEQUENCE:
			buf1 = ''
			for i in range(len(text_buf)):
				buf1 += text_buf[i]+' '
			#print(buf1)
			data_origin.append(buf1)
			label.append(line[2])
	print('data length : ', len(data_origin)) # 40000
	print('label length : ', len(label)) # 40000
	print('max len : ', max_len)
	print(data_origin[1])
	np.save(data_path+'train_data/data_origin.npy', data_origin)
	np.save(data_path+'train_data/label.npy', label)

if __name__ == '__main__':
	main()		



