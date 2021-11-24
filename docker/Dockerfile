ARG PYTORCH="1.7.0"
ARG CUDA="11.0"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx cmake ffmpeg\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN conda clean --all
WORKDIR /workspace

ENV FORCE_CUDA="1"
RUN pip install --upgrade pip
RUN pip install matplotlib==3.2.1
RUN pip install cython --no-cache-dir
RUN pip install opencv-contrib-python>=4.2.0.34
RUN pip install opencv-python
RUN pip install tqdm==4.45.0
RUN pip install numba==0.48
RUN pip install click typer requests tqdm pyspng ninja imageio imageio-ffmpeg==0.4.3
RUN pip install flask==1.1.2 flask_cors==3.0.9
RUN pip install flask_restful_swagger_2==0.35
RUN pip install dlib==19.22.1
RUN pip install tensorboard
RUN pip install jupyter

RUN pip install ftfy regex tqdm gdown
RUN pip install git+https://github.com/openai/CLIP.git

RUN pip install lpips
