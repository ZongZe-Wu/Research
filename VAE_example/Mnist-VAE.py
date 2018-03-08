
# coding: utf-8

# In[26]:


import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf
from ops import *
import time
import argparse,sys

# In[27]:


#importing data
from tensorflow.examples.tutorials.mnist import input_data
#one hot encoding returns an array of zeros and a single one. One corresponds to the class
data = input_data.read_data_sets("data/MNIST/", one_hot=True)


# In[28]:


print("Shape of images in training dataset {}".format(data.train.images.shape))
print("Shape of classes in training dataset {}".format(data.train.labels.shape))
print("Shape of images in testing dataset {}".format(data.test.images.shape))
print("Shape of classes in testing dataset {}".format(data.test.labels.shape))
print("Shape of images in validation dataset {}".format(data.validation.images.shape))
print("Shape of classes in validation dataset {}".format(data.validation.labels.shape))


# In[29]:


def batch_generation(data, label, batch_size):
	for i in range(0, data.shape[0], batch_size):
		yield data[i:i+batch_size], label[i:i+batch_size]


# In[32]:


class VAE(object):
	def __init__(self, name, input_size):
		self.input_size = input_size
		self.name = name
		self.n_latent = 2
		with tf.name_scope('inputs'):
			self.xs = tf.placeholder(tf.float32, shape = [None, self.input_size], name = 'xs')
			self.keep_prob = tf.placeholder(tf.float32,name = 'keep_prob')
		self.build_network()
	def sample_gumbel(self, shape, eps=1e-20): 
		"""Sample from Gumbel(0, 1)"""
		U = tf.random_uniform(shape,minval=0,maxval=1)
		return -tf.log(-tf.log(U + eps) + eps)

	def gumbel_softmax_sample(self, logits, temperature): 
		""" Draw a sample from the Gumbel-Softmax distribution"""
		y = logits + self.sample_gumbel(tf.shape(logits))
		return tf.nn.softmax( y / temperature)

	def gumbel_softmax(self, logits, temperature, hard=False):
		"""Sample from the Gumbel-Softmax distribution and optionally discretize.
		Args:
		logits: [batch_size, n_class] unnormalized log-probs
		temperature: non-negative scalar
		hard: if True, take argmax, but differentiate w.r.t. soft sample y
		Returns:
		[batch_size, n_class] sample from the Gumbel-Softmax distribution.
		If hard=True, then the returned sample will be one-hot, otherwise it will
		be a probabilitiy distribution that sums to 1 across classes
		"""
		y = self.gumbel_softmax_sample(logits, temperature)
		if hard:
			k = tf.shape(logits)[-1]
			#y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
			y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
			y = tf.stop_gradient(y_hard - y) + y
		return y
	def build_network(self, gumbel_softmax = False):
		print(self.name)
		with tf.name_scope(self.name):
			with tf.name_scope('Encoder'):
				with tf.variable_scope('conv1'):
					W_conv1 = weight_variables([3, 3, 1, 128], name = 'W_conv1')
					b_conv1 = bias_variables([128], name = 'b_conv1')
					h_conv1 = conv2d(tf.reshape(self.xs, [-1, 28, 28, 1]), W_conv1, 2, 'SAME') + b_conv1
					conv1 = lrelu(h_conv1)
					conv1 = dropout(conv1, self.keep_prob)
				with tf.variable_scope('conv2'):
					W_conv2 = weight_variables([3, 3, 128, 64], name = 'W_conv2')
					b_conv2 = bias_variables([64], name = 'b_conv2')
					h_conv2 = conv2d(conv1, W_conv2, 2, 'SAME') + b_conv2
					conv2 = lrelu(h_conv2)
					conv2 = dropout(conv2, self.keep_prob)
				with tf.variable_scope('conv3'):
					W_conv3 = weight_variables([3, 3, 64, 32], name = 'W_conv3')
					b_conv3 = bias_variables([32], name = 'b_conv3')
					h_conv3 = conv2d(conv2, W_conv3, 1, 'SAME') + b_conv3
					conv3 = lrelu(h_conv3)	
					conv3 = dropout(conv3, self.keep_prob)
				with tf.variable_scope('after_flatten'):
					flatten = tf.reshape(conv3, [-1, 7 * 7 * 32])
					W_conv4 = weight_variables([7 * 7 * 32, self.n_latent], name = 'W_conv4')
					b_conv4 = bias_variables([self.n_latent], name = 'b_conv4')
					self.z_mean = tf.matmul(flatten, W_conv4) + b_conv4
					W_conv5 = weight_variables([7 * 7 * 32, self.n_latent], name = 'W_conv5')
					b_conv5 = bias_variables([self.n_latent], name = 'b_conv5')
					self.z_log_sigma = tf.matmul(flatten, W_conv5) + b_conv5
					epsilon = tf.random_normal((tf.shape(flatten)[0], self.n_latent), 0, 1, dtype = tf.float32)
					self.z  = self.z_mean + tf.multiply(epsilon, tf.sqrt(tf.exp(self.z_log_sigma)))
				# 10
				# 28 * 28 * 1  784
			with tf.name_scope('Decoder'):
				with tf.variable_scope('dense'):
					W_conv1 = weight_variables([self.n_latent, 49], name = 'W_conv1')
					b_conv1 = bias_variables([49], name = 'b_conv1')
					h_dense = tf.matmul(self.z, W_conv1) + b_conv1
					dense = lrelu(h_dense)
					dense = tf.reshape(dense, [-1, 7, 7, 1])
				with tf.variable_scope('conv_tranpose1'):
					W_conv2 = weight_variables([4, 4, 64, 1], name = 'W_conv2')
					b_conv2 = bias_variables([64], name = 'b_conv2')
					h_conv2 = conv2d_transpose(dense, W_conv2, 2, [tf.shape(dense)[0], 14, 14, 64]) + b_conv2
					conv2 = lrelu(h_conv2)
					conv2 = dropout(conv2, self.keep_prob)
				with tf.variable_scope('conv_tranpose2'):
					W_conv3 = weight_variables([4, 4, 32, 64], name = 'W_conv3')
					b_conv3 = bias_variables([32], name = 'b_conv3')
					h_conv3 = conv2d_transpose(conv2, W_conv3, 2, [tf.shape(dense)[0], 28, 28, 32]) + b_conv3
					conv3 = lrelu(h_conv3)
					conv3 = dropout(conv3, self.keep_prob)
				with tf.variable_scope('conv_tranpose3'):
					W_conv4 = weight_variables([4, 4, 1, 32], name = 'W_conv4')
					b_conv4 = bias_variables([1], name = 'b_conv4')
					h_conv4 = conv2d_transpose(conv3, W_conv4, 1, [tf.shape(dense)[0], 28, 28, 1]) + b_conv4
					self.img = tf.reshape(tf.nn.sigmoid(h_conv4), [-1, self.input_size])
		with tf.name_scope('loss'):
			self.reconstruct_loss = -tf.reduce_sum(self.xs * tf.log(1e-10 + self.img) + (1-self.xs) * tf.log(1e-10 + 1 - self.img), 1)
			self.latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma - tf.square(self.z_mean) - tf.exp(self.z_log_sigma), 1)

			# weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) 
			# print("")
			# for w in weights:
			# 	shp = w.get_shape().as_list()
			# 	print("- {} shape:{} size:{}".format(w.name, shp, np.prod(shp)))
			# print("")
			# reg_ws = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
			# for w in reg_ws:
			# 	shp = w.get_shape().as_list()
			# 	print("- {} shape:{} size:{}".format(w.name, shp, np.prod(shp)))
			# print("")

			#self.loss = tf.reduce_mean(self.reconstruct_loss + self.latent_loss + 0.1*reg_ws)
			self.loss = tf.reduce_mean(self.reconstruct_loss + self.latent_loss)
		with tf.name_scope('train_op'):
			self._train_op = tf.train.AdamOptimizer(0.001).minimize(self.loss)


def show_images(images, cols = 5, titles = None):
	"""Display a list of images in a single figure with matplotlib.
	
	Parameters
	---------
	images: List of np.arrays compatible with plt.imshow.
	
	cols (Default = 1): Number of columns in figure (number of rows is 
						set to np.ceil(n_images/float(cols))).
	
	titles: List of titles corresponding to each image. Must have
			the same length as titles.
	"""
	assert((titles is None)or (len(images) == len(titles)))
	n_images = len(images)
	if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
	fig = plt.figure()
	for n, (image, title) in enumerate(zip(images, titles)):
		a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
		if image.ndim == 2:
			plt.gray()
		plt.imshow(image,cmap ='gray',aspect='auto',origin='lower')
		a.set_title(title)
	fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
	plt.show()
def show_images_2(x_sample, x_reconstruct):
	plt.figure(1,figsize=(8, 12))
	for i in range(5):
		plt.subplot(5, 2, 2*i + 1)
		plt.imshow(x_sample[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
		plt.title("Test input")
		plt.colorbar()
		plt.subplot(5, 2, 2*i + 2)
		plt.imshow(x_reconstruct[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
		plt.title("Reconstruction")
		plt.colorbar()
	plt.tight_layout()
def main():
	tf.reset_default_graph()
	sess = tf.Session()
	vae = VAE('VAE', data.train.images.shape[-1])
	sess.run(tf.global_variables_initializer())
	writer = tf.summary.FileWriter("logs/", sess.graph)
	saver = tf.train.Saver()
	training_loss_list = []
	batch_size = 32
	epoch  = 5000
	iteration = 1
	plt.ion()
	for i in range(epoch):
		for train_x, train_y in batch_generation(data.train.images, data.train.labels, batch_size):
			timestamp1 = time.time()
			feed_dict = {vae.xs: train_x, vae.keep_prob: 0.8}
			_, loss, img, z_mean = sess.run([vae._train_op, vae.loss, vae.img, vae.z_mean], feed_dict = feed_dict)
			timestamp2 = time.time()
			print('Epoch : ', i, 'iteration : ', iteration, '\ttime %.2f: ' % (timestamp2 - timestamp1), '\tLOSS : ', loss)
			training_loss_list.append(loss)
			iteration += 1
			if iteration % 400 == 0:
				plt.close()
				#show_images(np.reshape(img, [-1, 28, 28])) 
				show_images_2(train_x, img)
				plt.pause(1e-10)
				#plt.figure(2)
				#plt.plot(np.arange(len(training_loss_list)),training_loss_list)
				'''
				plt.figure(figsize=(8, 6)) 
				plt.scatter(z_mean[:, 0], z_mean[:, 1], c=np.argmax(train_y, 1))
				plt.colorbar()
				plt.grid()
				plt.pause(1e-10)
				'''
				print("SAVEEEEE MODELLLLLLL")
				saver.save(sess, "Model/model.ckpt")
				np.save('loss.npy', training_loss_list)
def test():
	tf.reset_default_graph()
	sess = tf.Session()
	vae = VAE('VAE', data.train.images.shape[-1])
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver()
	saver.restore(sess, "Model/model.ckpt")

	rand_index = np.random.choice(data.train.images.shape[-1] - 1, 5000)

	train_x = data.train.images[rand_index,:]
	train_y = data.train.labels[rand_index,:]
	feed_dict = {vae.xs: train_x, vae.keep_prob: 1.0}
	img, z_mean = sess.run([vae.img, vae.z_mean], feed_dict = feed_dict)
	#show_images_2(train_x, img)
	plt.figure(2, figsize=(8, 6)) 
	plt.scatter(z_mean[:, 0], z_mean[:, 1], c=np.argmax(train_y, 1), cmap='jet')
	plt.colorbar()
	plt.grid()
	plt.show()
	#plt.pause()
	
	nx = ny = 20
	x_values = np.linspace(-3, 3, nx)
	y_values = np.linspace(-3, 3, ny)
	batch_size = 32
	canvas = np.empty((28*ny, 28*nx))
	for i, yi in enumerate(x_values):
		for j, xi in enumerate(y_values):
			z_mean = np.array([[xi, yi]]*batch_size)
			feed_dict = {vae.z: z_mean, vae.keep_prob: 1.0}
			img = sess.run(vae.img, feed_dict = feed_dict)
			canvas[(nx-i-1)*28:(nx-i)*28, j*28:(j+1)*28] = img[0].reshape(28, 28)

	plt.figure(3, figsize=(8, 10))
	Xi, Yi = np.meshgrid(x_values, y_values)
	plt.imshow(canvas, origin="upper", cmap="gray")
	plt.tight_layout()
	plt.show()
if __name__ == '__main__':
	parser = argparse.ArgumentParser('')
	parser.add_argument('--train', action='store_true', help='whether train')
	parser.add_argument('--test', action='store_true', help='whether test')
	args = parser.parse_args()
	if args.train:
		main()	
	if args.test:
		test()
# In[ ]:




