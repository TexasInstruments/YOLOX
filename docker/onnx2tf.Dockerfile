# https://github.com/PINTO0309/onnx2tf/blob/main/Dockerfile
FROM ghcr.io/pinto0309/onnx2tf:1.8.1

# install OpenCV as root
USER root

RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1-mesa-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install opencv-python

# switch back to user
USER user
