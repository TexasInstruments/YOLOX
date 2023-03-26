FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

SHELL ["/bin/bash", "-c"]

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y \
    git \
    wget \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1-mesa-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN conda clean --all

# install dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

ENV PYTHONPATH=/work

WORKDIR /work
