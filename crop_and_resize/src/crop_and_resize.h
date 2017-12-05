void crop_and_resize_forward(
    THFloatTensor * image,
    THFloatTensor * boxes,      // [x1, y1, x2, y2]
    THIntTensor * box_index,    // range in [0, batch_size)
    const float extrapolation_value,
    const int crop_height,
    const int crop_width,
    THFloatTensor * crops
);