#!/bin/bash

pip install gdown
gdown --id 1aac2UaTDm-9h_PyI3hvZ6TZThcz6-Nol
unzip ACDC.zip
cp -r  ACDC/database/ database
rm -rf ACDC
rm -rf ACDC.zip

cd infrastucture
docker run --rm --gpus all -p 22:22 -p 8888:8888 -v $(dirname $(pwd)):/home/developer -w /home/developer bence1922/pytorch-jupyter-with-cuda:latest
