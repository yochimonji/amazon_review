FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

RUN apt update &&\
    apt install -y git

COPY requirements.txt ./
RUN pip install -r requirements.txt
