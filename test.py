import numpy as np
import torch
from torch.autograd import Variable
try:
    import tensorflow as tf
except ImportError:
    tf = None

from crop_and_resize.crop_and_resize import CropAndResizeFunction


def to_varabile(arr, requires_grad=False, is_cuda=True):
    tensor = torch.from_numpy(arr)
    if is_cuda:
        tensor = tensor.cuda()
    var = Variable(tensor, requires_grad=requires_grad)
    return var


def generate_data():
    batch_size = 10
    depth = 32
    im_height = 128
    im_width = 64
    n_boxes = 16

    # random rois
    xs = np.random.uniform(0, im_width, size=(n_boxes, 2)) / im_width
    ys = np.random.uniform(0, im_height, size=(n_boxes, 2)) / im_height
    xs.sort(axis=1)
    ys.sort(axis=1)

    boxes_data = np.stack((ys[:, 0], xs[:, 0], ys[:, 1], xs[:, 1]), axis=-1).astype(np.float32)
    box_index_data = np.random.randint(0, batch_size, size=n_boxes, dtype=np.int32)
    image_data = np.random.randn(batch_size, depth, im_height, im_width).astype(np.float32)

    return image_data, boxes_data, box_index_data


def compare_with_tf(image_data, boxes_data, box_index_data, crop_height, crop_width, is_cuda=True):

    # pytorch forward
    image_torch = to_varabile(image_data, requires_grad=True, is_cuda=is_cuda)

    boxes = to_varabile(boxes_data, requires_grad=True, is_cuda=is_cuda)
    box_index = to_varabile(box_index_data, requires_grad=False, is_cuda=is_cuda)
    crops_torch = CropAndResizeFunction(crop_height, crop_width, 0)(image_torch, boxes, box_index)
    crops_torch_data = crops_torch.data.cpu().numpy()

    # pytorch backward
    loss_torch = crops_torch.sum()
    loss_torch.backward()
    grad_torch_data = image_torch.grad.data.cpu().numpy()

    # tf forward & backward
    image_tf = tf.placeholder(tf.float32, (None, None, None, None), name='image')
    boxes = tf.placeholder(tf.float32, (None, 4), name='boxes')
    box_index = tf.placeholder(tf.int32, (None,), name='box_index')

    image_t = tf.transpose(image_tf, (0, 2, 3, 1))
    crops_tf = tf.image.crop_and_resize(image_t, boxes, box_index, (crop_height, crop_width))
    crops_tf = tf.transpose(crops_tf, (0, 3, 1, 2))

    loss_tf = tf.reduce_sum(crops_tf)
    grad_tf = tf.gradients(loss_tf, image_tf)[0]

    with tf.Session() as sess:
        crops_tf_data, grad_tf_data = sess.run(
            (crops_tf, grad_tf), feed_dict={image_tf: image_data, boxes: boxes_data, box_index: box_index_data}
        )

    crops_diff = np.abs(crops_tf_data - crops_torch_data)
    print('forward:', crops_tf_data.max(), crops_diff.min(), crops_diff.max(), crops_diff.mean())

    grad_diff = np.abs(grad_tf_data - grad_torch_data)
    print('backward:', grad_tf_data.max(), grad_diff.min(), grad_diff.max(), grad_diff.mean())


def test_backward(image_data, boxes_data, box_index_data, crop_height, crop_width):
    # TODO: gradient check

    print('end')


if __name__ == '__main__':
    def main():
        crop_height = 7
        crop_width = 10
        is_cuda = True

        image_data, boxes_data, box_index_data = generate_data()

        if tf is not None:
            compare_with_tf(image_data, boxes_data, box_index_data, crop_height, crop_width, is_cuda=is_cuda)
        else:
            print('without tensorflow')
        test_backward(image_data, boxes_data, box_index_data, crop_height, crop_width)

    main()
