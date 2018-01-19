import numpy as np
import torch
from torch import nn
from torch.autograd import Variable, gradcheck
from roi_align.crop_and_resize import CropAndResizeFunction
import matplotlib.pyplot as plt

def to_varabile(arr, requires_grad=False, is_cuda=True):
    tensor = torch.from_numpy(arr)
    if is_cuda:
        tensor = tensor.cuda()
    var = Variable(tensor, requires_grad=requires_grad)
    return var

crop_height = 50
crop_width = 50
is_cuda = True

# In this simple example the number of images and boxes is 2
img_path1 = '/path/to/first/img'
img_path2 = '/path/to/second/img'

# Define the boxes ( crops ) 
#box = [y1/heigth , x1/width , y2/heigth , x2/width]
box1 = np.array([ 0. , 0. , 1. , 1.]) # Takes all the image
box2 = np.array([ 0. , 0. , 1.0 , 0.5]) # Takes all the heigth and half width

# Create a batch of 2 boxes
boxes_data = np.stack([box1,box2],axis=0).astype(np.float32)
# Create an index to say which box crops which image
box_index_data = np.array([0,1]).astype(np.int32)

# Import the images from file
image_data1 = plt.imread(img_path1).transpose(2,0,1).astype(np.float32)*(1/255.)
image_data2 = plt.imread(img_path2).transpose(2,0,1).astype(np.float32)*(1/255.)

# Create a batch of 2 images
image_data = np.stack([image_data1,image_data2],axis=0)

# Convert from numpy to Variables
image_torch = to_varabile(image_data, is_cuda=is_cuda)
boxes = to_varabile(boxes_data, is_cuda=is_cuda)
box_index = to_varabile(box_index_data, is_cuda=is_cuda)

# Crops and resize bbox1 from img1 and bbox2 from img2
crops_torch = CropAndResizeFunction(crop_height, crop_width, 0)(image_torch, boxes, box_index)

# Visualize the crops
crops_torch_data = crops_torch.data.cpu().numpy().transpose(0,2,3,1)
fig = plt.figure()
plt.subplot(121)
plt.imshow(crops_torch_data[0])
plt.subplot(122)
plt.imshow(crops_torch_data[1])
plt.show()



