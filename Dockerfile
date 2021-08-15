FROM nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04

# Create user home directory
ENV USER_NAME wangmr
ENV HOME_DIR /home/$USER_NAME

# Replace HOST_UID/HOST_GUID with your user / group id
ENV HOST_UID 1006
ENV HOST_GID 1006

# Use bash as default shell, rather than sh
ENV SHELL /bin/bash
ENV DEBIAN_FRONTEND=noninteractive

COPY apt.txt /tmp/apt.txt
RUN apt -qq update && apt -qq install -y --no-install-recommends `cat /tmp/apt.txt` \
 && rm -rf /var/cache/*


WORKDIR ${HOME_DIR}

ENV CONDA_DIR ${HOME_DIR}/.conda

RUN wget -nv -O miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py38_4.9.2-Linux-x86_64.sh \
 && bash miniconda.sh -b -p ${CONDA_DIR} \
 && . ${CONDA_DIR}/etc/profile.d/conda.sh \
 && conda clean -y -a \
 && rm -rf miniconda.sh

ENV PATH ${CONDA_DIR}/bin:${PATH}

RUN conda install cmake -y && conda clean -y -a

COPY --chown=1006:1006 requirements.txt ${HOME_DIR}/requirements.txt
RUN conda install python==3.6.9 -y
RUN pip install -r requirements.txt --no-cache-dir


# Set up user

RUN apt-get update &&  apt-get install sudo && \
    apt install -y --no-install-recommends openssh-server vim && \
    cp /etc/sudoers /etc/sudoers.new && \
    echo "%docker ALL=(ALL:ALL) ALL" >> /etc/sudoers.new && \
    visudo -c -f /etc/sudoers.new && \
    cp /etc/sudoers.new /etc/sudoers && \
    rm /etc/sudoers.new
