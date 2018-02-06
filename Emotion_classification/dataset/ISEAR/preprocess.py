from meza import io

def main():

	records = io.read('isear_databank.mdb')
	file = open('preprocess.csv','w')
	buf = next(records)
	print(type(buf))
	while(buf):
		file.write(buf)
		buf = next(records)
	file.close()







if __name__ == '__main__':
	main()		