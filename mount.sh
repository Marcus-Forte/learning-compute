#/bin/sh
docker run -v.:/develop -t -d --gpus all --name compute nvcr.io/nvidia/cuda:12.2.2-devel-ubuntu22.04

#   ocl-icd-libopencl1 \
#     opencl-headers \
#     clinfo \
#     ;