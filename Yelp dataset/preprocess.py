import numpy as np
import json, sys
from pprint import pprint
import pandas as pd
from nltk.corpus import stopwords
positive_data = []
negative_data = []
neutral_data = []


def main(argv):
	# negative : 2748920/2741466, positive : 11588031/11575319
	file_positive = open('nostop_positive_sentence.csv','w')
	file_negative = open('nostop_negative_sentence.csv','w')
	pos_data = []
	neg_data = []
	stops = set(stopwords.words("english"))
	with open(argv[0]) as myfile:
		for line in myfile:
			buf = [word for word in line.strip().split(' ') if word not in stops]
			buf_list = " ".join(buf)
			if buf_list:
				pos_data.append(buf_list)
				file_positive.write(buf_list)
				file_positive.write('\n')
	with open(argv[1]) as myfile:
		for line in myfile:
			buf = [word for word in line.strip().split(' ') if word not in stops]
			buf_list = " ".join(buf)
			if buf_list:
				neg_data.append(buf_list)
				file_negative.write(buf_list)
				file_negative.write('\n')
	file_positive.close()
	file_negative.close()
	print('pos_data len: ', len(pos_data))
	print('neg_data len: ', len(neg_data))
	


if __name__ == '__main__':
	main(sys.argv[1:])



