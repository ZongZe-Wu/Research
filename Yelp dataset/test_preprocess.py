import numpy as np
import json, sys, time
from pprint import pprint
import pandas as pd
import re
import nltk
from textblob import TextBlob
positive_data = []
negative_data = []
neutral_data = []

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
	# buf = line.strip().split('\t')
	
	# do cleansing on hashtag, @, html, url
	# line = hashtag_pat.sub('', line)
	# line = at_pat.sub('',line)
	# line = html_pat.sub('',line)
	# line = url_pat.sub('',line)

	# dealing with wired spelling
	line = line.lower()
	line = re.sub('[{(_!@#$=;&/\-)}~:*?%`+]', ' ', line)
	line = line.replace('[', ' ')
	line = line.replace(']', ' ')
	line = re.sub('\d+', '_num_', line)
	line = re.sub("won't", 'will not', line)
	line = re.sub("n't", ' not', line)
	line = re.sub("'t", ' not', line)
	# line = re.sub("haven", 'have', line)
	# line = re.sub("don", ' do', line)
	line = re.sub("'s", " 's", line)
	line = re.sub("'m", ' am', line)
	line = re.sub("'ll", ' will', line)
	line = re.sub("'re", ' are', line)
	line = re.sub("'ve", ' have', line)
	line = re.sub("gonna", ' going to', line)
	line = re.sub("gotta", ' got to', line)
	line = re.sub("wanna", ' want to', line)
	line = re.sub("cannot", ' can not', line)
	line = re.sub("lemme", ' let me', line)
	line = re.sub("'", ' ', line)

	# do cleansing on emoji
	#line = emoji_pat.sub(' ', line)
	
	# clean multiple blankspaces
	line = re.sub('\s+', ' ', line).strip()
	
	return line
def write_file(words, file, text):
	if len(text) <= 10:
		for i in range(len(text)):
			if len(text[i]) > 3:
				if len(text[i].split()) <= 15 and len(text[i].split())>=2:
					file.write(text[i])
					file.write('\n')
def main(argv):
	words = set(nltk.corpus.words.words())
	it = 1
	data = []
	file_positive = open('positive_sentence.csv','w')
	file_negative = open('negative_sentence.csv','w')
	with open(argv[0]) as myfile:
		timestamp1 = time.time()
		for line in myfile:
			# print(line)
			json_buf = json.loads(line)
			text = json_buf['text'].replace('\n','')
			#print(text)
			star = json_buf['stars']
			# print(star)
			if len(text) > 3:
				b = TextBlob(text)
				if b.detect_language() != 'en':
					continue
			else:
				print(it, text)
				continue
			text_buf = [cleansing(x.strip()) for x in re.split('[!.?]', text) if x]
			# print(re.split('[!.?]', text))
			# print(text_buf)
			# print('\n')
			# print(text_buf)
			if star >= 3:
				write_file(words, file_positive, text_buf)
			elif star < 3:
				write_file(words, file_negative, text_buf)
			#if it % 10000 == 0:
			timestamp2 = time.time()
				# print()
			print(it, '\t', ' %.2f sentence per second' % (it/(timestamp2 - timestamp1)), end="\r")
			# if it > 100:
			# 	break
			
			#print(it)
			it += 1
	file_positive.close()
	file_negative.close()
	print('yo man')
if __name__ == '__main__':
	main(sys.argv[1:])



