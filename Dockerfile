FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

RUN apt-get update --fix-missing && apt-get install -y --no-install-recommends\
        build-essential \
        libfreetype6-dev \
        libpng-dev \
        libzmq3-dev \
        libspatialindex-dev \
        libsm6 \
        libgl1-mesa-dev \
        vim \
        git \
        curl \
        wget \
        zip \
        zsh \
        openssh-server \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get upgrade -y
RUN apt install -y software-properties-common

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3 get-pip.py
RUN pip3 install uv

RUN mkdir /kaggle
ENV PYTHONPATH="/kaggle:$PYTHONPATH"
ENV TBVACCINE=1

RUN mkdir /var/run/sshd
RUN echo 'root:we' | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -i 's/#PasswordAuthentication/PasswordAuthentication/' /etc/ssh/sshd_config
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd
ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

WORKDIR /kaggle
ENTRYPOINT ["/usr/sbin/sshd", "-D"]
