FROM dtr.collab.cra.com/proper-fm/cerulean:bamboo
# FROM continuumio/anaconda3:latest

LABEL maintainer 'mreposa@cra.com'

ENV DEBIAN_FRONTEND=noninteractive 

RUN mkdir /var/cerulean

COPY . /var/cerulean

WORKDIR /var/cerulean/

SHELL ["/bin/bash", "-c"]

RUN umask 000 && conda init bash \
    && source ~/.bashrc \
    && conda create -y --name proper-fm python=3.9 \
    && conda activate proper-fm \
    && python -m pip install -r requirements.txt \
    && conda deactivate

ENTRYPOINT conda init bash && /bin/bash
