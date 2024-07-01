# Use the NVIDIA CUDA base image
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# Add deadsnakes PPA to get Python 3.10
RUN apt-get update && apt-get install -y \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update

# Install Python 3.10 and dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as the default Python version
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Create and set the working directory
WORKDIR /workspace

# Copy siamesenet.py and data folder to the working directory
COPY siameseNet.py /workspace/
COPY ./data /workspace/data

# Install Python packages
RUN pip install torch torchvision numpy seaborn matplotlib scikit-learn

# Optional: Install additional packages
RUN pip install --upgrade pip && pip install -U torch torchvision numpy seaborn matplotlib scikit-learn

# Set the working directory to /workspace
WORKDIR /workspace

# Set the default command to open a bash shell
CMD ["bash"]
