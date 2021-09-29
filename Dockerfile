FROM nvidia/cuda:11.2.1-cudnn8-runtime-ubuntu18.04
ENV CONDALINK "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"

RUN apt-get update && apt-get install -y -q vim \
    git curl wget gcc make cmake g++ x11-xserver-utils \
    openjdk-8-jdk sudo xvfb ffmpeg zip unzip

ENV HOME=/root
RUN curl -so /root/miniconda.sh $CONDALINK \
 && chmod +x /root/miniconda.sh \
 && /root/miniconda.sh -b -p ~/miniconda \
 && rm /root/miniconda.sh
ENV PATH=/root/miniconda/bin:$PATH

RUN conda install conda-build \
 && conda create -y --name py37 python=3.7.3 \
 && conda clean -ya
ENV CONDA_DEFAULT_ENV=py37
ENV CONDA_PREFIX=/root/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH

RUN pip install  numpy scipy matplotlib jupyterlab scikit-learn \ 
    "ray[rllib]==1.6.0" pandas tensorflow==2.5.0 \
    wandb moviepy transformers gym==0.18.3 "aioredis<2.0" \
    tensorflow_probability==0.12.2 \
    docker==5.0.2 requests==2.26.0 tqdm==4.62.3 pyyaml==5.4.1
RUN conda install pytorch=1.8.1 torchvision cudatoolkit=11.1 -c pytorch -c nvidia
RUN pip install git+https://github.com/iglu-contest/iglu.git
RUN python -c 'import iglu;'
