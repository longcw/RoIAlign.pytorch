import numpy as np
import tensorflow as tf
import torch
from torch.autograd import Variable

from crop_and_resize.crop_and_resize import CropAndResizeFunction


def to_varabile(arr):
    return Variable(torch.from_numpy(arr))


def test_forward():
    batch_size = 10
    depth = 32
    im_height = 128
    im_width = 64
    crop_height = 7
    crop_width = 10
    n_boxes = 16

    # random rois
    xs = np.random.uniform(0, im_width, size=(n_boxes, 2)) / im_width
    ys = np.random.uniform(0, im_height, size=(n_boxes, 2)) / im_height
    xs.sort(axis=1)
    ys.sort(axis=1)

    boxes_data = np.stack((ys[:, 0], xs[:, 0], ys[:, 1], xs[:, 1]), axis=-1).astype(np.float32)
    box_index_data = np.random.randint(0, batch_size, size=n_boxes, dtype=np.int32)
    image_data = np.random.randn(batch_size, depth, im_height, im_width).astype(np.float32)

    # pytorch forward
    image = to_varabile(image_data)
    boxes = to_varabile(boxes_data)
    box_index = to_varabile(box_index_data)
    crops = CropAndResizeFunction(crop_height, crop_width, 0)(image, boxes, box_index)

    crops_torch = crops.data.cpu().numpy()

    # tf forward
    with tf.Session() as sess:
        image = tf.placeholder(tf.float32, (None, depth, im_height, im_width), name='image')
        boxes = tf.placeholder(tf.float32, (None, 4), name='boxes')
        box_index = tf.placeholder(tf.int32, (None,), name='box_index')

        image_t = tf.transpose(image, (0, 2, 3, 1))
        crops_op = tf.image.crop_and_resize(image_t, boxes, box_index, (crop_height, crop_width))
        crops_op = tf.transpose(crops_op, (0, 3, 1, 2))

        crops_tf = sess.run(crops_op, feed_dict={image: image_data, boxes: boxes_data, box_index: box_index_data})

    diff = np.abs(crops_tf - crops_torch)
    print(diff.min(), diff.max(), diff.mean())
    print('end')


if __name__ == '__main__':
    test_forward()
