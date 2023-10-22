# Docker Quick Start

## Build the image
git clone https://github.com/drigba/medical_image_segmentation.git
cd infrastructure
docker login
docker build . -t pytorch-jupyter-with-cuda

## Push the container 
Your teammates can use your version. (We use docker hub, please create an account)
docker image push docker.io/bence1922/pytorch-jupyter-with-cuda:latest 

## Start the container
docker run --rm --gpus all -p 22:22 -p 8888:8888 -v $(dirname $(pwd)):/home/developer -w /home/developer bence1922/pytorch-jupyter-with-cuda:latest
in the log you can see the token for jupyterlab

## Jupyter lab
http://localhost:8888
You can add this as a new kernel and choose "Existing Jupyter server"

### Use docker in WSL
Dowload Docker Desktop and start it. Enable to using it in WSL, then the docker engine can be used in your WSL.
