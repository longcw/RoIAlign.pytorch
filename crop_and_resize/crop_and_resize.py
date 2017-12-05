import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from _ext import crop_and_resize as _backend


class CropAndResizeFunction(Function):

    def __init__(self, crop_height, crop_width, extrapolation_value=0):
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.extrapolation_value = extrapolation_value

    def forward(self, image, boxes, box_ind):
        crops = torch.zeros_like(image)
        _backend.crop_and_resize_forward(
            image, boxes, box_ind,
            self.extrapolation_value, self.crop_height, self.crop_width, crops)

        # save for backward
        self.im_size = image.size()
        self.save_for_backward(boxes, box_ind)

        return crops

    def backward(self, grad_outputs):
        boxes, box_ind = self.saved_tensors

        grad_image = grad_outputs.clone().resize_(*self.im_size).zero_()
        _backend.crop_and_resize_backward(
            grad_outputs.clone(), boxes, box_ind, grad_image    # .clone() because a strange bug in pytorch
        )

        return grad_image, None, None

