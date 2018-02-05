import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random
import re




def main():
	data_path = '/home/zong-ze/Research/Emotion_classification/dataset/Emotion_Intensity/train/'
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
					buf = line.strip().split('\t')
					buf[-1] = re.sub('[!@#$.=;&/\-]', ' ', buf[-1])
					anger_data.append(buf)
			elif it == 1:
				print(it)
				for line in myfile:
					buf = line.strip().split('\t')
					buf[-1] = re.sub('[!@#$.=;&/\-]', ' ', buf[-1])
					fear_data.append(buf)
			elif it == 2:
				print(it)
				for line in myfile:
					buf = line.strip().split('\t')
					buf[-1] = re.sub('[!@#$.=;&/\-]', ' ', buf[-1])
					joy_data.append(buf)
			else:
				print(it)
				for line in myfile:
					buf = line.strip().split('\t')
					buf[-1] = re.sub('[!@#$.=;&/\-]', ' ', buf[-1])
					sadness_data.append(buf)			
			it += 1
	anger_data_array = np.array(anger_data)
	print('anger_data_array shape : ', anger_data_array.shape)
	fear_data_array = np.array(fear_data)
	print('fear_data_array shape : ', fear_data_array.shape)
	joy_data_array = np.array(joy_data)
	print('joy_data_array shape : ', joy_data_array.shape)
	sadness_data_array = np.array(sadness_data)
	print('sadness_data_array shape : ', sadness_data_array.shape)
	






if __name__ == '__main__':
	main()		


