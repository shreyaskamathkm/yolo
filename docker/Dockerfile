ARG BASE_IMG=nvidia/cuda:12.6.2-devel-ubuntu22.04
FROM ${BASE_IMG}

ARG USERNAME=user
ARG WORKDIR=/home/${USERNAME}/yolo

# Set shell to bash and ensure pipefail is on for safer builds
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV CUDA_HOME=/usr/local/cuda

# Environment Variables
ENV DEBIAN_FRONTEND=noninteractive \
    PATH=${PATH}:${CUDA_HOME}/bin \
    LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CUDA_HOME}/lib64

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    curl \
    gcc \
    git \
    make \
    wget \
    zlib1g \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3-dev \
    python3-pip \
    python-is-python3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create user first so we can install pip packages to their local path
RUN useradd -m -s /bin/bash -G sudo ${USERNAME} \
    && echo "${USERNAME}:${USERNAME}" | chpasswd \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER ${USERNAME}
WORKDIR ${WORKDIR}

# Install PyTorch with the correct CUDA 12.4/12.6 compatible index
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cu126

# Clone and install the repo
# We use --user to ensure permissions are correct in the home directory
RUN git clone https://github.com/shreyaskamathkm/yolo.git . \
    && pip install --no-cache-dir --user .[export]

RUN echo 'export PATH=$PATH:${HOME}/.local/bin' >> ~/.bashrc
RUN source ~/.bashrc
