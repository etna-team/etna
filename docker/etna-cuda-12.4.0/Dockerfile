ARG BASE_IMAGE=nvidia/cuda:12.4.0-base-ubuntu22.04
ARG DISTRO_ARCH=ubuntu2204/x86_64
ARG CUDA_VERSION_FOR_TORCH=cu124

FROM ${BASE_IMAGE}

ARG CUDA_VERSION_FOR_TORCH
ARG DISTRO_ARCH

RUN apt-get -y update \
    && apt-get install --no-install-recommends  -y software-properties-common curl git openssh-client \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get -y update \
    && apt-get --no-install-recommends  -y install build-essential python3.10 python3.10-distutils python3.10-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python get-pip.py \
    && rm get-pip.py

COPY requirements.txt /
RUN pip install --no-cache-dir -r requirements.txt -f https://download.pytorch.org/whl/${CUDA_VERSION_FOR_TORCH}/torch_stable.html \
    && pip install --no-cache-dir prophet \
    && rm -rf ~/.cache
WORKDIR /code

CMD [ "bash" ]