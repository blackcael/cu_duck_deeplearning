FROM ros:noetic-ros-base-focal

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-lc"]

RUN sed -i 's|http://archive.ubuntu.com/ubuntu|https://archive.ubuntu.com/ubuntu|g' /etc/apt/sources.list \
    && sed -i 's|http://security.ubuntu.com/ubuntu|https://security.ubuntu.com/ubuntu|g' /etc/apt/sources.list

RUN apt-get -o Acquire::Retries=6 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 update \
    && apt-get -o Acquire::Retries=6 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 install -y --no-install-recommends \
    python3-pip \
    python3-pygame \
    joystick \
    evtest \
    nano \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Convenience: source ROS in interactive shells.
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc
