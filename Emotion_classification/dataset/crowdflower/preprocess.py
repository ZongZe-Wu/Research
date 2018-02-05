import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random
import re



def main():
	data_path = '/home/zong-ze/Research/Emotion_classification/dataset/crowdflower/'
	data = []
	# read raw data
	with open(data_path+'emotion_text.csv') as myfile:
		next(myfile)
		for line in myfile:
			buf = line.strip().split(',')
			for i in range(len(buf)):
				buf[i] = buf[i].replace('"', '')
			if len(buf) > 4:
				a = ''
				for j in range(3, len(buf)):
					a += buf[j]
				buf = buf[:3] + [a]
			buf[-1] = buf[-1].lower()
			buf[-1] = re.sub('[{(_!@#$.=;&/\-)}~:*?%`]', ' ', buf[-1])
			buf[-1] = buf[-1].replace('[', ' ')
			buf[-1] = buf[-1].replace(']', ' ')
			buf[-1] = re.sub("]", ' ', buf[-1])
			buf[-1] = re.sub("'t", ' not', buf[-1])
			buf[-1] = re.sub("don", ' do', buf[-1])
			buf[-1] = re.sub("'s", ' is', buf[-1])
			buf[-1] = re.sub("'m", ' am', buf[-1])
			buf[-1] = re.sub("'ll", ' will', buf[-1])
			buf[-1] = re.sub("'re", ' are', buf[-1])
			buf[-1] = re.sub("'ve", ' have', buf[-1])
			buf[-1] = re.sub("gonna", ' going to', buf[-1])
			buf[-1] = re.sub("gotta", ' got to', buf[-1])
			buf[-1] = re.sub("wanna", ' want to', buf[-1])
			buf[-1] = re.sub("cannot", ' can not', buf[-1])
			buf[-1] = re.sub("lemme", ' let me', buf[-1])
			buf[-1] = re.sub("'", ' ', buf[-1])
			data.append(buf)
	data_array = np.array(data)
	print('data shape : ', data_array.shape)
	data_dict = {}
	id2sentence = {}
	#data_dict = {data_array[i,1] : data_array[i,0] for i in range(data_array.shape[0])}
	for i in range(data_array.shape[0]):
		if data_array[i,1] not in data_dict:
			data_dict[data_array[i,1]] = [data_array[i,0]]
		else:
			data_dict[data_array[i,1]].append(data_array[i,0])
		id2sentence[data_array[i,0]] = data_array[i,3]

	for k, v in data_dict.items():
		print(k,'\t',len(v),'\n',id2sentence[v[random.randint(0,len(v)-1)]])
		if k == 'anger':
			for i in range(len(v)):
				print(i,'\t',v[i])
		file = open('preprocess_data/'+str(k)+'.csv','w')
		for id_val in v:
			file.write(id2sentence[id_val]) 
			file.write('\n')
		file.close()




if __name__ == '__main__':
	main()		


