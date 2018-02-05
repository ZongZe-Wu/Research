import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random


def main():
	data_path = '/home/zong-ze/Research/Emotion_classification/dataset/EmoBank/corpus/'
	data = []
	# read raw data
	with open(data_path+'raw.tsv') as myfile:
		next(myfile)
		for line in myfile:
			data.append(line.strip().split('\t'))
			#print(data)
	data_array = np.array(data)
	print('data shape : ', data_array.shape)

	data_dict = {data_array[i,0] : data_array[i,1] for i in range(data_array.shape[0])}
	print(data_dict['Acephalous-Cant-believe_4_47'])

	# read VAD data
	vad_data = []
	with open(data_path+'writer.tsv') as myfile:
		next(myfile)
		for line in myfile:
			vad_data.append(line.strip().split('\t'))

	vad_data_array = np.array(vad_data)
	print('vad_data_array shape : ', vad_data_array.shape)
	vad_data_dict = {vad_data_array[i,0] : vad_data_array[i,1:].astype(float) for i in range(vad_data_array.shape[0])}
	print(vad_data_dict['Acephalous-Cant-believe_4_47'])
	#print(vad_data_dict)
	#sentence_vad_value = {k : vad_data_dict.get(k, 'empty') for k, v in data_dict.items()}
	sentence_vad_value = {}
	for k, v in data_dict.items():
		if k in vad_data_dict:
			sentence_vad_value[k] = vad_data_dict.get(k)
	# if not vad_data_dict.get(k)
	#print(sentence_vad_value)
	print(sentence_vad_value['Acephalous-Cant-believe_4_47'])
	#vad_inv_map = {v: k for k, v in vad_data_dict.items()}
	#print('vad_inv_map len : ', len(vad_inv_map))
	# plot vad_data_array
	'''
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	x = (vad_data_array[:, 1]).astype(float) # Arousal
	y = (vad_data_array[:, 3]).astype(float) # Valence
	z = (vad_data_array[:, 2]).astype(float) # Dominance
	#plt.scatter(x, y)
	ax.scatter(x, y, z, alpha=0.8, c='green', edgecolors='none', s=30)
	ax.set_xlabel('Arousal')
	ax.set_ylabel('Valence')
	ax.set_zlabel('Dominance')
	plt.show()
	'''
	# a : 1.5~4.5, v : 1 ~5 
	# high arousal, high valence : happiness - elation
	ha_hv = []
	# high arousal, low valence : hostility - anger 
	ha_lv = [] 
	# low arousal, high valence : relaxed - clam
	la_hv = []
	# low arousal, low valence : bored - sadness
	la_lv = []
	# mid arousal, mid valence : netural
	ma_mv = []

	for k, v in sentence_vad_value.items():
		#if v != 'empty':
		if v[0] > 3.2 and v[2] > 3.2:
			ha_hv.append(k)
		elif v[0] > 3.2 and v[2] < 2.8:
			ha_lv.append(k)
		elif v[0] < 2.8 and v[2] > 3.2:
			la_hv.append(k)
		elif v[0] < 2.8 and v[2] < 2.8:
			la_lv.append(k)
		else:
			ma_mv.append(k)

	print('ha_hv amount : ', len(ha_hv))
	print('ha_lv amount : ', len(ha_lv))
	print('la_hv amount : ', len(la_hv))
	print('la_lv amount : ', len(la_lv))
	print('ma_mv amount : ', len(ma_mv))

	print('happiness - elation example : ', data_dict[ha_hv[random.randint(0,len(ha_hv)-1)]])
	print('hostility - anger  example : ', data_dict[ha_lv[random.randint(0,len(ha_lv)-1)]])
	print('relaxed - clam example : ', data_dict[la_hv[random.randint(0,len(la_hv)-1)]])
	print('bored - sadness example : ', data_dict[la_lv[random.randint(0,len(la_lv)-1)]])
if __name__ == '__main__':
	main()		