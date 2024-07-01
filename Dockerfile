FROM nvidia/cuda:12.3.2-devel-ubuntu22.04


ENV USER abiel
ENV PASS dummypassword123
ENV UID 1000
ENV GID 1000

RUN apt-get update && apt-get install -y sudo vim

RUN groupadd -g ${GID} ${USER}
RUN useradd -g ${GID} -u ${UID} -p ${PASS} -m ${USER} && \
    echo ${USER}:${PASS} | chpasswd && \
    adduser ${USER} sudo
USER ${USER}

WORKDIR /home/${USER}/workspace
RUN echo ${PASS}| sudo -S apt install -y python3 python-is-python3 pip