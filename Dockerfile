FROM ubuntu:22.04

RUN apt update
RUN apt install -y python3-pip

COPY ./requirements.txt /requirements.txt
RUN pip install -r /requirements.txt
