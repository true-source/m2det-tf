import json
import argparse
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from m2det import M2det
from m2det.data import Loader
from m2det.losses import calc_loss


argparser = argparse.ArgumentParser(description='Train m2det')
argparser.add_argument(
	'-c',
	'--config',
	help='path to configuration(json) file', required=True)
args = argparser.parse_args()

config = json.load(open(args.config, "r"))
data_loader = Loader(config)

inputs = Input(shape=(320, 320, 3,))
outputs = M2det(config).forward(inputs)

model = Model(inputs=inputs, outputs=outputs)
optim = Adam() 
for epoch in range(config['train']['nb_epochs']):
	epoch_loss_avg = tf.metrics.Mean()
	# epoch_accuracy = tf.metrics.Accuracy()
	for x, y_ in data_loader.batches():
		with tf.GradientTape(watch_accessed_variables=False) as tape:
			tape.watch(model.trainable_variables)
			y = model(x)
			loss = calc_loss(y, y_)
		grads = tape.gradient(loss, model.trainable_variables)
		optim.apply_gradients(zip(grads, model.trainable_variables))
		epoch_loss_avg(loss)
		# epoch_accuracy(tf.argmax(y, axis=1, output_type=tf.float32), y_)
	# end of epoch
	print("Epoch {:03d}: Loss: {:.3f}".format(epoch, epoch_loss_avg.result()))
	if epoch == 50:
		tf.saved_model.save(model, config['train']['model_dir'])