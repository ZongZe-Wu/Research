import numpy as np
import matplotlib.pyplot as plt
import random
import re
from mpl_toolkits.mplot3d import Axes3D

def cleansing(line):
	# define patterns
	'''
	unicode for fxxking emojis:
	https://unicode.org/emoji/charts/full-emoji-list.html
	https://unicode.org/Public/emoji/5.0/emoji-data.txt
		
	reference pages:
	https://gist.github.com/jinstrive/34d1796bd2dd46b6aa52
	http://blog.csdn.net/dszgf5717/article/details/52071883
	https://marcobonzanini.com/2015/03/09/mining-twitter-data-with-python-part-2/
	'''
	hashtag_pat = re.compile(r'(?:\#+[\w_]+[\w\'_\-]*[\w_]+)')
	at_pat = re.compile(r'(?:@[\w_]+)')
	html_pat = re.compile(r'<[^>]+>')
	url_pat = re.compile(r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+')
	emoji_pat = re.compile(r'[\W]')
	#emoji_pat = re.compile(u'[\U00010000-\U0010ffff] | [\U00002000-\U000027BF]')
	#emoji_pat = re.compile(u'([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])')
	
	# after strip() and split() each buf has four element: [id, text, emotion, emotion_ratio]
	buf = line.strip().split('\t')
	
	# do cleansing on hashtag, @, html, url
	buf[1] = hashtag_pat.sub('', buf[1])
	buf[1] = at_pat.sub('',buf[1])
	buf[1] = html_pat.sub('',buf[1])
	buf[1] = url_pat.sub('',buf[1])

	# dealing with wired spelling
	buf[1] = buf[1].lower()
	buf[1] = re.sub('[{(_!@#$.=;&/\-)}~:*?%`+]', ' ', buf[1])
	buf[1] = buf[1].replace('[', ' ')
	buf[1] = buf[1].replace(']', ' ')
	buf[1] = re.sub("'t", ' not', buf[1])
	buf[1] = re.sub("don", ' do', buf[1])
	buf[1] = re.sub("'s", ' is', buf[1])
	buf[1] = re.sub("'m", ' am', buf[1])
	buf[1] = re.sub("'ll", ' will', buf[1])
	buf[1] = re.sub("'re", ' are', buf[1])
	buf[1] = re.sub("'ve", ' have', buf[1])
	buf[1] = re.sub("gonna", ' going to', buf[1])
	buf[1] = re.sub("gotta", ' got to', buf[1])
	buf[1] = re.sub("wanna", ' want to', buf[1])
	buf[1] = re.sub("cannot", ' can not', buf[1])
	buf[1] = re.sub("lemme", ' let me', buf[1])
	buf[1] = re.sub("'", ' ', buf[1])

	# do cleansing on emoji
	buf[1] = emoji_pat.sub(' ', buf[1])

	return buf

def labeling(emotion):
	# define dict
	emotion_list = {'anger': 0, 'fear': 1, 'joy': 2, 'sadness': 3}
	
	return emotion_list[emotion]

def main():
	data_path = './train/'
	emo_path = ['anger-ratings-0to1.train.txt', 'fear-ratings-0to1.train.txt', 'joy-ratings-0to1.train.txt', 'sadness-ratings-0to1.train.txt']
	anger_data, fear_data, joy_data, sadness_data = [], [], [], []

	# read raw data
	it = 0
	
	for emo in emo_path:
		with open(data_path+emo) as myfile:
			print(data_path+emo)
			if it == 0:
				print(it)
				for line in myfile:
					buf = cleansing(line)
					buf[2] = labeling(buf[2])
					anger_data.append(buf)
					
			elif it == 1:
				print(it)
				for line in myfile:
					buf = cleansing(line)
					buf[2] = labeling(buf[2])
					fear_data.append(buf)
						
			elif it == 2:
				print(it)
				for line in myfile:
					buf = cleansing(line)
					buf[2] = labeling(buf[2])
					joy_data.append(buf)
					
			else:
				print(it)
				for line in myfile:
					buf = cleansing(line)
					buf[2] = labeling(buf[2])
					sadness_data.append(buf)
			it += 1

	# check
	anger_data_array = np.array(anger_data)
	print('anger_data_array shape : ', anger_data_array.shape)
	print(anger_data_array[0:30])
	print("-----------------------------------------------------")

	fear_data_array = np.array(fear_data)
	print('fear_data_array shape : ', fear_data_array.shape)
	print(fear_data_array[0:30])
	print("-----------------------------------------------------")

	joy_data_array = np.array(joy_data)
	print('joy_data_array shape : ', joy_data_array.shape)
	print(joy_data_array[0:30])
	print("-----------------------------------------------------")
		
	sadness_data_array = np.array(sadness_data)
	print('sadness_data_array shape : ', sadness_data_array.shape)
	print(sadness_data_array[0:30])
	print("-----------------------------------------------------")

	# save training file
	text = np.vstack((anger_data_array, fear_data_array, joy_data_array, sadness_data_array))
	print("-----------------------------------------------------")
	print(text.shape)
	print("-----------------------------------------------------")
	np.save('./train/training_data.npy', text)

if __name__ == '__main__':
	main()


