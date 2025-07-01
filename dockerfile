# Sử dụng image PyTorch chính thức
FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel

# Cập nhật hệ thống và cài đặt các công cụ cần thiết
RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget \
    curl \
    git \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgl1-mesa-dri \
    libglapi-mesa \
    build-essential \
    python3-dev \
    g++ \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt Python 3.7 từ deadsnakes PPA
RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y python3.7 python3.7-dev python3.7-venv && \
    rm -rf /var/lib/apt/lists/*

# Cài đặt pip cho Python 3.7
RUN wget https://bootstrap.pypa.io/pip/3.7/get-pip.py -O get-pip.py && \
    python3.7 get-pip.py && \
    rm get-pip.py

# Tạo và kích hoạt môi trường ảo
RUN python3.7 -m venv /root/vit_env
ENV PATH="/root/vit_env/bin:$PATH"

# Cập nhật pip
RUN pip install --upgrade pip

# Cài đặt các thư viện Python từ requirements.txt
COPY requirements.txt .
RUN pip install -r requirements.txt

# Cài đặt kernel cho Jupyter
RUN pip install ipykernel && \
    python -m ipykernel install --user --name vit_env --display-name "vit_env"

# Cấu hình Jupyter Notebook
ENV JUPYTER_PORT=8888
EXPOSE 8888

# Khởi động Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]