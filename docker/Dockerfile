ARG BASE_IMG=nvidia/cuda:12.6.2-devel-ubuntu22.04
FROM ${BASE_IMG}

ARG USERNAME=user
ARG WORKDIR=/home/${USERNAME}/yolo

SHELL ["/bin/bash", "-c"]


ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${PATH}:${CUDA_HOME}/bin
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CUDA_HOME}/lib64

RUN apt-get update \
    && apt-get install -y \
    sudo \
    curl \
    gcc \
    git \
    make \
    wget \
    zlib1g \
    protobuf-compiler \
    libgl1-mesa-dev \
    graphviz \
    python-is-python3 \
    python3-pip \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

# Make user
RUN echo "root:root" | chpasswd \
    && useradd \
    --create-home \
    --home-dir /home/${USERNAME} \
    --shell /bin/bash \
    --user-group \
    --groups adm,sudo \
    ${USERNAME} \
    && echo "${USERNAME}:${USERNAME}" | chpasswd \
    && cat /dev/null > /etc/sudoers.d/${USERNAME} \
    && echo "%${USERNAME}    ALL=(ALL)   NOPASSWD:    ALL" >> \
    /etc/sudoers.d/${USERNAME} \
    && mkdir -p ${WORKDIR} \
    && chown ${USERNAME}:${USERNAME} ${WORKDIR}

USER ${USERNAME}
WORKDIR ${WORKDIR}

RUN git clone https://github.com/shreyaskamathkm/yolo.git .

# Install CUDA-specific torch before the rest so pip install . doesn't
# pull in a CPU build from PyPI
RUN pip install --no-cache-dir \
    torch==2.6.0 \
    torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu126

RUN pip install --no-cache-dir .

# Ensure pip-installed packages are available in the PATH
RUN echo 'export PATH=${PATH}:${HOME}/.local/bin' >> ~/.bashrc

# Optional: Source .bashrc to apply changes in the current session
RUN source ~/.bashrc
