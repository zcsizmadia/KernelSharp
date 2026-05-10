FROM mcr.microsoft.com/vscode/devcontainers/base:ubuntu-24.04

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

ARG DOTNET_VERSIONS=8.0 9.0 10.0

ARG CUDA_VERSION=13.2
ARG CUDA_VERSION_DASH=${CUDA_VERSION//./-}
ARG CUDA_OS=ubuntu2404

# Install prerequisites
RUN apt-get update && apt-get install -y \
    wget \
    sudo \
    gnupg \
    software-properties-common \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# --- CUDA Installation ---
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/${CUDA_OS}/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && rm cuda-keyring_1.1-1_all.deb

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/${CUDA_OS}/x86_64/cuda-${CUDA_OS}.pin \
    && mv cuda-${CUDA_OS}.pin /etc/apt/preferences.d/cuda-repository-pin-600

RUN add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/${CUDA_OS}/x86_64/ /" \
    && apt-get update \
    && apt-get -y install cuda-${CUDA_VERSION_DASH} \
    && rm -rf /var/lib/apt/lists/*

# --- .NET Installation (8, 9, 10) ---
# Download the official install script and run for each version
RUN curl -sSL https://dot.net/v1/dotnet-install.sh -o dotnet-install.sh \
    && chmod +x dotnet-install.sh \
    && for version in ${DOTNET_VERSIONS}; do \
        ./dotnet-install.sh --channel $version; \
    done \
    && rm dotnet-install.sh

# --- Environment Variables ---
# CUDA Paths
ENV PATH="/usr/local/cuda-${CUDA_VERSION}/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda-${CUDA_VERSION}/lib64:${LD_LIBRARY_PATH}"

# .NET Paths (Default install location is ~/.dotnet)
ENV DOTNET_ROOT="/root/.dotnet"
ENV PATH="${DOTNET_ROOT}:${DOTNET_ROOT}/tools:${PATH}"

CMD ["/bin/bash"]