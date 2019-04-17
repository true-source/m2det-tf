import tensorflow as tf
from tensorflow.keras.layers import Conv2D, UpSampling2D

class FFM:
	def __init__(self, features1, features2):
		self.features1 = features1
		self.features2 = features2

	def v1(self):
		out1 = Conv2D(kernel_size=(1, 1), filters=256)(self.features1)
		out2 = Conv2D(kernel_size=(1, 1), filters=512)(self.features2)
		out2 = UpSampling2D(size=(2, 2))(out2)	
		out = tf.concat([out1, out2], -1)
		return out

	def v2(self, features3):
		v1_out = self.v1()
		out = Conv2D(kernel_size=(1, 1), filters=128)(v1_out)
		out = tf.concat([out, features3], -1)
		return out