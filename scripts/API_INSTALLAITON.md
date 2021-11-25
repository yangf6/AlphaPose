## Step 1:
Fork mmpose

## Step 2:
Add Flask endpoint  
example: https://github.com/yangf6/AlphaPose/blob/pose-api/scripts/demo_api.py#L391

## Step 3:
Install it on VM   
Git clone your mmpose repo   
Install mmpose, and all other required package  
Copy model file into VM  
```
ssh -i linda-access-key.pem azureuser@{public-ip-address}
ssh -i {you-key}.pem azureuser@{public-ip-address}

export AZURE_STORAGE_CONNECTION_STRING=“”, you can get from the yogamvp 


export FLASK_APP=scripts/demo_api
export FLASK_ENV=development

flask run --host=0.0.0.0 --port=8080 (run it in the foreground)
```

## Step 4: optional 
Put in the docker file, and dockerize it  
example Dockerfile with cuda as base image:
```
FROM nvidia/cuda:11.2.0-cudnn8-devel-ubuntu18.04

## install base paackage
RUN apt-get update && apt-get install -y \
        software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y \
        python3.7 \
        python3-pip
RUN apt-get update && apt-get install -y \
        python3-distutils \
        python3-setuptools
RUN apt-get install -y python3.7-dev


RUN rm -f /usr/bin/python 
RUN ln -s /usr/bin/python3.7 /usr/bin/python

RUN python -m pip install pip --upgrade pip

RUN apt-get -y install git
RUN apt-get -y install libyaml-dev
RUN apt-get install -y libgl1-mesa-dev
RUN apt-get install ffmpeg libsm6 libxext6 -y
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Pacific
RUN apt-get install -y python3-tk
RUN python -m pip install azureml-core azure-storage-blob azureml-defaults azureml-mlflow

## install torch and torchvision
RUN python -m pip install --upgrade pip
RUN python -m pip install --upgrade Pillow
RUN python -m pip install opencv-python
RUN python -m pip install pandas
RUN python -m pip install numpy
RUN python -m pip install setuptools wheel
RUN python -m pip install "mxnet<2.0.0"
RUN python -m pip install autogluon
RUN python -m pip install azure-kusto-data
RUN python -m pip install azure-kusto-ingest


```

## Step 5: expose container port if dockerize it
