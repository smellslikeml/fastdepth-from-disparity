FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
          git \
                cmake \
                build-essential \
                python3-setuptools \
                curl \
                wget \
                gnupg2 \
                lsb-release \
                ca-certificates \
                python3-pip \
                python3-dev \
                libopenblas-dev \
                libopenmpi-dev \
                openmpi-bin \
                openmpi-common \
                gfortran \
                libomp-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN apt-get update
RUN apt-get install -y libhdf5-serial-dev hdf5-tools ffmpeg

WORKDIR /install
RUN wget https://download.pytorch.org/whl/cu92/torch-0.4.1-cp36-cp36m-linux_x86_64.whl
RUN wget https://download.pytorch.org/whl/torchvision-0.2.0-py2.py3-none-any.whl
RUN pip3 install --upgrade pip
RUN pip3 install h5py matplotlib imageio scikit-image opencv-python
RUN pip3 install torch-0.4.1-cp36-cp36m-linux_x86_64.whl
RUN pip3 install torchvision-0.2.0-py2.py3-none-any.whl
RUN pip3 install blobconverter

RUN pip3 uninstall -y scipy 
RUN pip3 install scipy==1.2.2
WORKDIR /app
