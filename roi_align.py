import torch
from torch import nn

from crop_and_resize.crop_and_resize import CropAndResizeFunction


class RoIAlign(nn.Module):

    def __init__(self, crop_height, crop_width, extrapolation_value=0):
        super(RoIAlign, self).__init__()

        self.crop_height = crop_height
        self.crop_width = crop_width
        self.extrapolation_value = extrapolation_value

    def forward(self, featuremap, boxes, box_ind):
        """
        RoIAlign based on crop_and_resize.
        See more details on https://github.com/ppwwyyxx/tensorpack/blob/6d5ba6a970710eaaa14b89d24aace179eb8ee1af/examples/FasterRCNN/model.py#L301
        :param featuremap: NxCxHxW
        :param boxes: Mx4 float box with (x1, y1, x2, y2) **without normalization**
        :param box_ind: M
        :return: MxCxoHxoW
        """
        x1, y1, x2, y2 = torch.split(boxes, 4, dim=1)

        spacing_w = (x2 - x1) / float(self.crop_width)
        spacing_h = (y2 - y1) / float(self.crop_height)

        image_height, image_width = featuremap.size()[2:4]
        nx0 = (x1 + spacing_w / 2 - 0.5) / float(image_width - 1)
        ny0 = (y1 + spacing_h / 2 - 0.5) / float(image_height - 1)

        nw = spacing_w * float(self.crop_width - 1) / float(image_width - 1)
        nh = spacing_w * float(self.crop_height - 1) / float(image_height - 1)

        boxes = torch.cat((ny0, nx0, ny0 + nh, nx0 + nw), 1)

        return CropAndResizeFunction(self.crop_height, self.crop_width, self.extrapolation_value)(featuremap, boxes, box_ind)