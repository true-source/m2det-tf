import sys
import json
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
import numpy as np
from m2det.feature_extractors import VGG16, Darknet21
from m2det import FFM
from m2det import TUM
from m2det import SFAM
from m2det.utils import bilinear_upsampler
from m2det.box_predictors import simple_predictor
from m2det.classifier import simple_classifier
from m2det.data import Loader
from m2det.losses import calc_loss
tf.enable_eager_execution()


config_file = sys.argv[1]
assert config_file, "Specify config file"
config = json.load(open(config_file, "r"))

loader = Loader(config)

for x_batch, y_batch in loader.batches():
	print (x_batch.shape)

print ("done")
imgs = tf.Variable(imgs, dtype=tf.float32)

# f1, f2 = Darknet21(imgs, config).forward()
f1, f2 = VGG16(imgs, config).forward()
ffm = FFM(f1, f2)

#collecting decoder outputs from TUMs
decoder_outs = []
for i in range(config["model"]["tums_no"]):
	if i == 0:
		features = ffm.v1()
		features = Conv2D(kernel_size=(1, 1), strides=(1, 1), filters=256)(features)
	else:
		features = ffm.v2(decoder_outs[i-1][-1])
	decoder_outs.append(TUM(config,features).forward())

#constructing mlfpn using SFAM
mlfpn = SFAM(config, decoder_outs).forward()
boxes = []
classes = []
for feature_cube in mlfpn:
	box_pred = simple_predictor(config, feature_cube)
	cls_pred = simple_classifier(config, feature_cube)
	boxes.append(box_pred)
	classes.append(cls_pred)


all_box = tf.concat(boxes, axis=1)
all_classes = tf.concat(classes, axis=1)
all_box = tf.reshape(all_box, [-1, int(all_box.shape[1]/4), 4])
y_pred = tf.concat([all_box,all_classes], axis=2)


lbls = tf.to_float(lbls)
loss = calc_loss(lbls, y_pred)
print (loss)