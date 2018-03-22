#!/usr/bin/env bash

CUDA_PATH=/usr/local/cuda

echo "Compiling crop_and_resize kernels by nvcc..."
cd roi_align/src/cuda
$CUDA_PATH/bin/nvcc -c -o crop_and_resize_kernel.cu.o crop_and_resize_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52

cd ../../../roi_align
python build.py

cd ..
python setup.py install
find  $CONDA_PREFIX -name roi_align | awk '{mkdir $0 "/_ext" }' |bash
find  $CONDA_PREFIX -name roi_align | awk '{print "cp -r roi_align/_ext/* " $0 "/_ext/" }' |bash
