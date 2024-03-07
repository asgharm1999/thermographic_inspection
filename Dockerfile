# Use the official Ubuntu base image
FROM ubuntu:22.04

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gnupg2 \
    lsb-release \
    python3 \
    python3-pip

# Add ROS2 GPG key and repository
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS2 Humble
RUN apt-get update && apt-get install -y \
    ros-humble-desktop \
    python3-rosdep

# Initialize rosdep
RUN rosdep init && rosdep update

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Copy the project files to the working directory
COPY . .

# Source the ROS2 setup file
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# Specify the command to run your Python script
CMD ["python3", "preprocess/rosbagstest.py"]