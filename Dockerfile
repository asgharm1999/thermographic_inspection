# Stage 1: Base image with Ubuntu 22.04
FROM ubuntu:22.04 AS base

# Set DEBIAN_FRONTEND to noninteractive to avoid getting stuck on prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gnupg2 \
    lsb-release \
    sudo

# Add ROS 2 repository
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Stage 2: Install ROS-Humble-Desktop and dependencies
FROM base AS ros

# Set up retry function and alternative archives
ENV DEBIAN_FRONTEND=noninteractive
RUN echo "apt-get update && apt-get install -y --allow-unauthenticated \$@" > /usr/local/bin/apt-install && \
    chmod +x /usr/local/bin/apt-install && \
    echo "deb http://archive.ubuntu.com/ubuntu/ $(lsb_release -cs) main restricted universe multiverse" > /etc/apt/sources.list && \
    echo "deb http://archive.ubuntu.com/ubuntu/ $(lsb_release -cs)-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb http://archive.ubuntu.com/ubuntu/ $(lsb_release -cs)-security main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb http://mirror.math.princeton.edu/pub/ubuntu/ $(lsb_release -cs) main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb http://mirror.math.princeton.edu/pub/ubuntu/ $(lsb_release -cs)-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb http://mirror.math.princeton.edu/pub/ubuntu/ $(lsb_release -cs)-security main restricted universe multiverse" >> /etc/apt/sources.list

# Install ROS-Humble-Desktop and ros-dev-tools with retry and fallback
RUN apt-get update && \
    apt-install ros-humble-desktop ros-dev-tools || \
    (sleep 30 && apt-get update && apt-install ros-humble-desktop ros-dev-tools)

# Stage 3: Install project dependencies
FROM ros AS project

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file to the working directory
COPY requirements.txt .

# Install Python and pip
RUN apt-get install -y python3 python3-pip

# Source the ROS setup file
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# Install project dependencies with retry and fallback, ignoring not found packages
RUN while read requirement; do \
    pip3 install --no-cache-dir --timeout 10 --retries 1 --default-timeout 10 $requirement \
    --find-links https://pypi.org/simple/ --trusted-host pypi.org || \
    # --find-links https://www.piwheels.org/simple --trusted-host www.piwheels.org \
    # --find-links https://pypi.python.org/simple/ --trusted-host pypi.python.org || \
    echo "Skipping package $requirement"; \
    done < requirements.txt


# Copy the project files to the working directory
#COPY . .


# Reset DEBIAN_FRONTEND environment variable
ENV DEBIAN_FRONTEND=

# Specify the command to run your Python script
# CMD ["bash", "-c", "source ~/.bashrc && python3 preprocess/rosbagstest.py"]
