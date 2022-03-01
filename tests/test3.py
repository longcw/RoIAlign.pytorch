import torch
import sys,os
sys.path.append(os.getcwd())
from roi_align import RoIAlign      # RoIAlign module
from roi_align import CropAndResize # crop_and_resize module

# input feature maps (suppose that we have batch_size==2)
image = torch.arange(0., 49).view(1, 1, 7, 7).repeat(2, 1, 1, 1)
image[0] += 10
print('image: ', image)


# for example, we have two bboxes with coords xyxy (first with batch_id=0, second with batch_id=1).
boxes = torch.Tensor([[1, 0, 5, 4],
                     [0.5, 3.5, 4, 7]])

box_index = torch.tensor([0, 1], dtype=torch.int) # index of bbox in batch

# RoIAlign layer with crop sizes:
crop_height = 4
crop_width = 4
roi_align = RoIAlign(crop_height, crop_width)

# make crops:
crops = roi_align(image, boxes, box_index)

print('crops:', crops)