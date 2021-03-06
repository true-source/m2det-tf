import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def bilinear_upsampler(tensor, new_shape):
	return tf.image.resize(tensor, new_shape)


#Reference : https://github.com/tadax/m2det/blob/master/utils/generate_priors.py
def generate_anchors(num_scales=3, anchor_scale=2.0, image_size=320, shapes=[40, 20, 10, 5, 3, 1]):
	anchor_configs = {}
	for shape in shapes:
		anchor_configs[shape] = []
		for scale_octave in range(num_scales):
			for aspect_ratio in [(1, 1), (1.41, 0.71), (0.71, 1.41)]:
				anchor_configs[shape].append(
					(
						image_size / shape,
						scale_octave / float(num_scales),
						aspect_ratio
					)
				)

	boxes_all = []
	for _, configs in anchor_configs.items():
		boxes_level = []
		for config in configs:
			stride, octave_scale, aspect = config
			base_anchor_size = anchor_scale * stride * (2 ** octave_scale)
			anchor_size_x_2 = base_anchor_size * aspect[0] / 2.0
			anchor_size_y_2 = base_anchor_size * aspect[1] / 2.0
			x = np.arange(stride / 2, image_size, stride)
			y = np.arange(stride / 2, image_size, stride)
			xv, yv = np.meshgrid(x, y)
			xv = xv.reshape(-1)
			yv = yv.reshape(-1)
			boxes = np.vstack(
				(
					yv - anchor_size_y_2, xv - anchor_size_x_2,
					yv + anchor_size_y_2, xv + anchor_size_x_2)
				)
			boxes = np.swapaxes(boxes, 0, 1)
			boxes_level.append(np.expand_dims(boxes, axis=1))
		boxes_level = np.concatenate(boxes_level, axis=1)
		boxes_level /= image_size
		boxes_all.append(boxes_level.reshape([-1, 4]))

	anchor_boxes = np.vstack(boxes_all)

	return anchor_boxes


def clip_boxes(boxes, img_shape=(320, 320)):
	"""Clip boxes to image boundaries."""
	# x1 >= 0
	boxes[:, 0] = np.maximum(boxes[:, 0], 0)
	# y1 >= 0
	boxes[:, 1] = np.maximum(boxes[:, 1], 0)
	# x2 < im_width
	boxes[:, 2] = np.minimum(boxes[:, 2], img_shape[0] - 1)
	# y2 < img_height
	boxes[:, 3] = np.minimum(boxes[:, 3], img_shape[1] - 1)
	return boxes


def iou(anchors, boxes):
	if len(boxes.shape) == 2:
		boxes = boxes[0,:]
	if len(anchors.shape) == 1:
		anchors = anchors.reshape(1,-1)
	x1 = np.maximum(anchors[:, 0], boxes[0])
	y1 = np.maximum(anchors[:, 1], boxes[1])
	x2 = np.minimum(anchors[:, 2], boxes[2])
	y2 = np.minimum(anchors[:, 3], boxes[3])

	width = (x2 - x1)
	height = (y2 - y1)
	area_overlap = np.maximum(width*height, 0)
	l = (anchors[:, 2] - anchors[:, 0])
	b = (anchors[:, 3] - anchors[:, 1])
	area_an = l * b
	l = (boxes[2] - boxes[0])
	b = (boxes[3] - boxes[1])
	area_bx = l * b
	union = np.maximum((area_an + area_bx) - area_overlap,0)
	iou = area_overlap / (union)

	#setting iou to 0 which doesn't have overlap with ground truth
	iou[iou == np.inf] = 0
	iou[iou == -np.inf] = 0
	iou[anchors[:,2] < boxes[0]] = 0
	iou[anchors[:,0] > boxes[2]] = 0
	iou[anchors[:,3] < boxes[1]] = 0
	iou[anchors[:,1] > boxes[3]] = 0



	return iou


def nms(dets, confidence, thresh=0.4):
	outputs = []
	while len(dets) > 0:
		ix = np.argmax(confidence)
		result = dets[ix]
		outputs.append(result)
		dets = np.delete(dets, ix, axis=0)
		box1 = np.expand_dims(result, axis=0)
		to_delete = []
		for jx in range(len(dets)):
			box2 = np.expand_dims(dets[jx], axis=0)
			iou_ = iou(box1, box2)
			if iou_ >= thresh:
				to_delete.append(jx)
		for jx in to_delete[::-1]:
			dets = np.delete(dets, jx, axis=0)
	return np.array(outputs, dtype=np.float32)


def visualize(img, boxes):
	if len(boxes.shape) == 1:
		boxes = boxes.reshape(1,-1)
	fig, ax = plt.subplots(1)
	ax.imshow(img)
	boxes_xy_wh = boxes
	boxes_xy_wh[:, 0] = boxes[:, 0]
	boxes_xy_wh[:, 1] = boxes[:, 1]
	boxes_xy_wh[:, 2] = boxes[:, 2] - boxes[:, 0]
	boxes_xy_wh[:, 3] = boxes[:, 3] - boxes[:, 1]
	for i in range(len(boxes_xy_wh)):
		k = 0
		if i == 0:
			ec = 'r'
		else:
			ec = 'g'
		s = patches.Rectangle(
			(boxes_xy_wh[i][k], boxes_xy_wh[i][k+1]),
			boxes_xy_wh[i][k+2], boxes_xy_wh[i][k+3],
			linewidth=1, edgecolor=ec,
			facecolor="none"
			)
		ax.add_patch(s)
	plt.show()


if __name__ == "__main__":
	gt_box = np.array([[0, 0, 99, 99]])
	boxes = np.array(
		[[0, 0, 74, 74],
		[0, 0, 24, 24],
		[0, 0, 70, 70],
		[0, 0, 50, 50],
		[10, 20, 60, 70]]
		)
	ans = np.array(
		[0.5, 0.062, 0.5, 0.25]
		)
	iou_ = iou(boxes, gt_box)
	check = iou_[:4]/ans
	print (iou_)
	print("iou cases %s"%(np.greater(check, 0.9)))
	nms_box = nms(boxes, iou_)
	print ("nms cases are %s"%(np.allclose(nms_box, boxes[:2, :])))
