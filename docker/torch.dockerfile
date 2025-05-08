FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel
LABEL authors="Fabian Schotte"

RUN apt-get update && \
    apt-get -y install openssh-server && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

WORKDIR /notebooks
