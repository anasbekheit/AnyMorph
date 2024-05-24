# Base image with CUDA and Ubuntu 20.04
FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel

ARG uid
ARG user
ARG cuda

# CUDA includes
ENV CUDA_PATH /usr/local/cuda
ENV CUDA_INCLUDE_PATH /usr/local/cuda/include
ENV CUDA_LIBRARY_PATH /usr/local/cuda/lib64

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
# to avoid dialogues for tzada install
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y --allow-unauthenticated --no-install-recommends \
    build-essential apt-utils cmake git curl vim ca-certificates \
    libjpeg-dev libpng-dev python3.6 python3-pip python3-setuptools \
    libgtk3.0 libsm6 python3-venv cmake ffmpeg pkg-config \
    qtbase5-dev libqt5opengl5-dev libassimp-dev libpython3.6-dev \
    libboost-python-dev libtinyxml-dev bash python3-tk \
    wget unzip libosmesa6-dev software-properties-common \
    libopenmpi-dev libglew-dev graphviz graphviz-dev patchelf

RUN pip3 install pip --upgrade

#RUN add-apt-repository ppa:jamesh/snap-support && apt-get update && apt install -y patchelf
RUN rm -rf /var/lib/apt/lists/*

# For some reason, I have to use a different account from the default one.
# This is absolutely optional and not recommended. You can remove them safely.
# But be sure to make corresponding changes to all the scripts.

WORKDIR /$user
RUN chmod -R 777 /$user
RUN chmod -R 777 /usr/local

RUN useradd -d /$user -u $uid $user

RUN pip install --upgrade pip
RUN pip install pymongo
RUN pip install numpy scipy pyyaml matplotlib ruamel.yaml networkx tensorboardX pygraphviz
RUN pip install torch-scatter==2.0.3 -f https://pytorch-geometric.com/whl/torch-1.4.0+cu101.html
RUN pip install torch-sparse==0.6.1 -f https://pytorch-geometric.com/whl/torch-1.4.0+cu101.html
RUN pip install torch-geometric
RUN pip install "tensorflow<2"
RUN pip install gym==0.13.1
RUN pip install gym[atari]
RUN pip install pybullet cffi
RUN pip install seaborn
RUN pip install git+https://github.com/yobibyte/pgn.git
RUN pip install six beautifulsoup4 termcolor num2words
RUN pip install lxml tabulate coolname lockfile glfw
RUN pip install "Cython<3"
RUN pip install sacred
RUN pip install imageio
RUN pip install xmltodict
RUN pip install torchfold

USER $user
RUN mkdir -p /$user/.mujoco \
    && wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d /$user/.mujoco \
    && mv /$user/.mujoco/mujoco200_linux /$user/.mujoco/mujoco200 \
    && rm mujoco.zip \
    && wget https://www.roboti.us/file/mjkey.txt -O /$user/.mujoco/mjkey.txt

ENV LD_LIBRARY_PATH /$user/.mujoco/mujoco200/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
ENV MUJOCO_PY_MUJOCO_PATH /$user/.mujoco/mujoco200
ENV MUJOCO_PY_MJKEY_PATH /$user/.mujoco/mjkey.txt
RUN pip3 install mujoco_py==2.0.2.8
RUN pip3 install baselines==0.1.5 --user

WORKDIR /$user/amorpheus
ENV PYTHONPATH /$user/amorpheus:/$user/amorpheus/modular-rl/src:/$user/amorpheus/modular-rl
