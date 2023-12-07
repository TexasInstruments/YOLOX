#!/usr/bin/env bash

# Copyright (c) 2018-2021, Texas Instruments
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

######################################################################

# system packages
#sudo apt-get install -y libjpeg-dev zlib1g-dev cmake libffi-dev protobuf-compiler

######################################################################
# upgrade pip
pip3 install --no-input --upgrade pip setuptools

######################################################################
echo "installing pytorch - use the applopriate index-url from https://pytorch.org/get-started/locally/"
pip3 install --no-input torch==2.0.1+cu118 torchvision==0.15.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html

echo 'Installing python packages...'
# there as issue with installing pillow-simd through requirements - force it here
#pip uninstall --yes pillow
pip install pillow
pip install --no-input -U --force-reinstall pillow-simd
pip3 install --no-input cython wheel numpy
pip3 install --no-input torchinfo pycocotools opencv-python

echo "installing requirements"
pip3 install --no-input -r requirements.txt

######################################################################
echo "Installing mmcv"
pip3 install --no-input mmcv-full==1.4.8 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.1/index.html

######################################################################
# building onnx from soure requires carefull steps
# make sure that we are using system cmake
pip uninstall --yes cmake
# pybind11[global] is needed for building the onnx package.
# for some reason, this has to be installed before the requirements file is used.
pip3 install --no-input pybind11[global] protobuf
pybind11_DIR=$(pybind11-config --cmakedir) pip3 install --no-input https://github.com/TexasInstruments/onnx/archive/tidl-j7.zip

######################################################################
echo 'installing the python package...'
python3 setup.py develop

