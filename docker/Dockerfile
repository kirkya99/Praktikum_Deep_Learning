FROM tensorflow/tensorflow:2.19.0-gpu-jupyter
LABEL authors="Fabian Schotte"

RUN apt-get update
RUN apt-get -y install cudnn-cuda-12
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

WORKDIR /notebooks