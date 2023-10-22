# Docker Quick Start Guide

## Step 1: Build the Docker Image

1. Clone the project repository:
   ```bash
   git clone https://github.com/drigba/medical_image_segmentation.git
   ```

2. Navigate to the 'infrastructure' directory:
   ```bash
   cd infrastructure
   ```

3. Log in to Docker (if not already logged in):
   ```bash
   docker login
   ```

4. Build the Docker image with the desired tag:
   ```bash
   docker build -t pytorch-jupyter-with-cuda .
   ```

## Step 2: Push the Docker Container

To share your Docker container with your teammates, you'll need to upload it to Docker Hub. If you haven't created a Docker Hub account, please do so.

1. Push the container to Docker Hub:
   ```bash
   docker push docker.io/bence1922/pytorch-jupyter-with-cuda:latest
   ```

## Step 3: Start the Docker Container

Your teammates can now use your version of the container. Start the container with the following command:
```bash
docker run --rm --gpus all -p 22:22 -p 8888:8888 -v $(dirname $(pwd)):/home/developer -w /home/developer bence1922/pytorch-jupyter-with-cuda:latest
```

You'll find the token for JupyterLab in the container's log output.

## Step 4: Access JupyterLab

1. Open your web browser and visit the following URL to access JupyterLab:
   ```
   http://localhost:8888
   ```

2. You can add this container as a new kernel and choose "Existing Jupyter server."

## Using Docker in WSL

If you're using Windows Subsystem for Linux (WSL), follow these steps:

1. Download Docker Desktop and install it.

2. Start Docker Desktop.

3. Enable WSL integration to make the Docker engine accessible from your WSL environment.

## Docker container

This Dockerfile is designed to create a containerized environment for running machine learning tasks using PyTorch with CUDA 11.7 and cuDNN 8 runtime. It includes the following key steps:

1. It starts from the base image `pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime`.

2. The Dockerfile updates the package list and installs the following necessary packages:
   - `openssh-server`
   - `libgl1`
   - `tmux`
   - `byobu`
3. It defines environment variables for the user's home directory, infrastructure path, and a default CSV file path.
4. The Dockerfile copies the project files into the image.
5. It exposes ports 22 and 8888 for SSH and JupyterLab access.
6. The Dockerfile upgrades `pip` and installs Python packages listed in `requirements.txt`.
7. It creates a 'developer' user with the password 'password'.
8. The default shell for the 'developer' user is set to `/bin/bash`.
9. The Dockerfile defines an entry point that starts the SSH service and launches JupyterLab with specific configurations, allowing remote access to JupyterLab with support for running machine learning tasks.
