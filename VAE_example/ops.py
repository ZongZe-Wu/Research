import tensorflow as tf
import tensorflow.contrib.layers as layers


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