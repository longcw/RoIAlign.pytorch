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


## Usage
