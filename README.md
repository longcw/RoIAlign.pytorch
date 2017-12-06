# RoIAlign for PyTorch
This is a PyTorch version of [RoIAlign](https://arxiv.org/abs/1703.06870).
This implementation is based on `crop_and_resize` 
and supports both forward and backward on CPU and GPU.


## Introduction
The `crop_and_resize` function is ported from [tensorflow](https://www.tensorflow.org/api_docs/python/tf/image/crop_and_resize),
and has the same interface with tensorflow version, except the input feature map 
should be in `NCHW` order in PyTorch. 
They also have the same output value (error < 1e-5) for both forward and backward as we expected, 
see the comparision in `test.py`.

The `RoIAlign` is a wrap of `crop_and_resize` 
that uses boxes with unnormalized `(x1, y1, x2, y2)` as input.
See more details in [tensorpack](https://github.com/ppwwyyxx/tensorpack/blob/6d5ba6a970710eaaa14b89d24aace179eb8ee1af/examples/FasterRCNN/model.py#L301).


## Usage
+ Build and test
    ```bash
    chmod a+x make.sh
    ./make.sh
    python test.py
    ```

+ Use RoIAlign or crop_and_resize
    ```python
    from roi_align.roi_align import RoIAlign, CropAndResize  
  
    # input data
    image = to_varabile(image_data, requires_grad=True, is_cuda=is_cuda)
    boxes = to_varabile(boxes_data, requires_grad=False, is_cuda=is_cuda)
    box_index = to_varabile(box_index_data, requires_grad=False, is_cuda=is_cuda)
    
    # RoIAlign layer
    roi_align = RoIAlign(crop_height, crop_width)
    crops = roi_align(image, boxes, box_index)
    ```