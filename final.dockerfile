#!/bin/bash
# Cập nhật hệ thống và cài đặt các công cụ cần thiết
apt-get update && apt-get install -y \
    software-properties-common \
    wget \
    curl \
    git \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0

# Cài đặt Python 3.7
add-apt-repository ppa:deadsnakes/ppa -y
apt-get updated
apt-get install -y python3.7 python3.7-dev python3.7-venv

# Cài đặt pip cho Python 3.7
wget https://bootstrap.pypa.io/pip/3.7/get-pip.py -o get-pip.py
python3.7 get-pip.py

# Tạo môi trường ảo với Python 3.7
python3.7 -m venv /root/vit_env
source /root/vit_env/bin/activate

# Cài đặt Jupyter để hỗ trợ Jupyter Notebook
pip install jupyter

# Khởi động Jupyter Notebook
if [ -n "$JUPYTER_PORT" ]; then
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root &
fi

# Docker Repository And Environment
image path:
pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel

ssh -p 40520 root@83.10.126.17 -L 8080:localhost:8080

# rsync upload project to vast
rsync -av --progress --partial --no-compress -e "ssh -p 40520 -i /home/anhlee/.ssh/id_rsa" /mnt/e/NCKH/Nutrition/ root@83.10.126.17:/workspace/


# CONVERT LOCAL TO REMOTE
ssh -p 21406 -i /home/anhlee/.ssh/id_rsa root@50.7.159.181 -L 8080:localhost:8080

# TEST MMCV
python -c "from mmcv.ops import bbox_overlaps; print('mmcv._ext loaded successfully')"


# TEST segment_anything
python -c "import segment_anything"

# create kernel
pip install ipykernel
python -m ipykernel install --user --name vit_env --display-name "vit_env"

# install package-----------------------------------------------------------------

apt-get update
apt-get install -y libgl1-mesa-glx
apt-get install -y libgl1 libgl1-mesa-dri libglapi-mesa
apt-get update

apt-get install -y build-essential python3.7-dev g++ ninja-build

pip install --upgrade pip
pip cache purge

pip install mmcv-full==1.3.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html
pip cache purge
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip cache purge

# First install
pip install tensorflow==1.13.1
pip install keras==2.2.4
pip cache purge


# Second install
pip install pandas==0.24.2

# Third install
pip install timm==0.6.12

# Forth install
pip install terminaltables==3.1.10
pip install fvcore==0.1.5.post20221221
pip install cloudpickle==2.2.1
pip install omegaconf==2.3.0
pip install pycocotools==2.0.6
pip cache purge


# Fifth install
pip install scikit-learn==0.21.1
pip install scikit-image==0.16.2
pip install matplotlib==3.0.3
pip install pyntcloud==0.1.2
pip install pythreejs==2.1.1
pip install ipython
pip install Flask==1.1.1
pip install fuzzywuzzy==0.18.0
pip install xlrd==1.2.0
pip cache purge

# sixth install
pip install jupyter notebook terminado

# seventh install
pip install git+https://github.com/facebookresearch/segment-anything.git@6fdee8f

# last install
pip install fastapi
pip install uvicorn
pip install python-multipart
pip install pyngrok
pip cache purge


pip uninstall -y protobuf
pip install protobuf==3.20.3
pip uninstall -y scipy
pip install scipy==1.2.
pip uninstall -y h5py
pip install h5py==2.10.0
pip install httpx
pip cache purge

