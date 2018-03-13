import tensorflow as tf
import tensorflow.contrib.layers as layers
import matplotlib.pyplot as plt
import numpy as np

def weight_variables(shape, name, trainable = True):
	initializer = tf.random_normal_initializer(mean = 0., stddev = 0.02 )
	return tf.get_variable(shape = shape, initializer = initializer, name = name, trainable = trainable) 
def bias_variables(shape, name, trainable = True):
	initializer = tf.constant_initializer(0.0)
	return tf.get_variable(shape = shape, initializer = initializer, name = name, trainable = trainable)
def conv2d(x, W, stride, padding_type):
	return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding_type)
def conv2d_transpose(x, W, stride, output_shape):
	return tf.nn.conv2d_transpose(x, W, output_shape, [1,stride, stride, 1])
#def conv2d_transpose(x, num_output, kernel, stride, activatoin_fn)
#	return layers.conv2d_transpose(x, num_output, kernel_size=kernel, stride=stride, activatoin_fn=activatoin_fn)
def max_pool(x, k, l, padding_type):
	return tf.nn.max_pool(x, ksize=[1, k, 1, l],strides=[1, 1, 1, 1], padding=padding_type)
def dropout(x, dropout_keep_prob):
	return tf.nn.dropout(x, dropout_keep_prob)
def lrelu(x, leak=0.2, name="lrelu"):
	#leak = tf.Variable(leak, name = 'leak')
	#return tf.maximum(x, tf.scalar_mul(leak,x))
	return tf.maximum(x, leak*x)
def lrelu_batch_norm(x, leak=0.2, name="lrelu_bn"):
	x = tf.contrib.layers.batch_norm(x)
	return tf.maximum(x, leak*x)
'''	
def lrelu(features, alpha=0.2, name=None):
	with ops.name_scope(name, "LeakyRelu", [features, alpha]):
		features = ops.convert_to_tensor(features, name="features")
		alpha = ops.convert_to_tensor(alpha, name="alpha")
	return math_ops.maximum(alpha * features, features)
'''	
def relu(x):
	return tf.nn.relu(x)
def relu_batch_norm(x):
	return tf.nn.relu(tf.contrib.layers.batch_norm(x))
def plot_confusion_matrix(cm, classes,
						  normalize=False,
						  title='Confusion matrix',
						  cmap=plt.cm.Blues):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')

	print(cm)

	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt),
				 horizontalalignment="center",
				 color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	
def sample_gumbel(shape, eps=1e-20): 
		"""Sample from Gumbel(0, 1)"""
		U = tf.random_uniform(shape,minval=0,maxval=1)
		return -tf.log(-tf.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature): 
	""" Draw a sample from the Gumbel-Softmax distribution"""
	y = logits + sample_gumbel(tf.shape(logits))
	return tf.nn.softmax( y / temperature)

def gumbel_softmax(logits, temperature, hard=False):
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
	y = gumbel_softmax_sample(logits, temperature)
	if hard:
		k = tf.shape(logits)[-1]
		#y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
		y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
		y = tf.stop_gradient(y_hard - y) + y
	return y