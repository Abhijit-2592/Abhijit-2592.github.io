---
layout: post # needs to be post
title: Building a personal deep learning environment with GPU acceleration for computer vision with Docker # title of your post
featured-img: docker_logo #optional - if you want you can include hero image
date: 2018-06-22
published: true
---
## Introduction:
In my first post, I would like to share my personal build environment which I use for my computer vision projects. But, why [Docker](https://www.docker.com/)? That too on a PC? Well, in short: It is easy to manage :P. Lets say you are building a development environment on your host machine and you install a new pacakge and suddenly your OS is wrecked! well... good luck with it. To solve this, docker provides a **light weight**, **isolated** and **reproducable**  environment. If something goes wrong remove the image and build again with a single command VOILA! Ok No more small talks, lets dive into building the environment. We will build the environment in **Ubuntu-16.04**


**TLDR** : All the codes to setup the environment are present in my github repo [dockerfiles](https://github.com/Abhijit-2592/dockerfiles). Feel free to clone and use it.


## My host machine:
I use [acer predator helios-300](https://www.acer.com/ac/en/US/content/predator-helios300-series) with **Ubuntu-16.04** installed. This laptop is a steal for a price tag of approximately `$1000`. It is highly recommended and economical for personal deep learning projects! and well obviously gaming :P.

**Hardware specs**:
* GPU - NVIDIA GTX-1060 - 6GB memory
* RAM - 16GB
* Disk space - 1TB (firecuda)
* Processor - Intel i7 - 7th generation


## Breaking the ice:
First things first. We need to install [Docker CE](https://docs.docker.com/install/linux/docker-ce/ubuntu/#install-docker-ce). Next we need to install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker). It is a thin wrapper around Docker from Nvidia since, Docker Engine does not natively support NVIDIA GPUs with containers.

We are going to follow a two step process to setup the environment
1. First we will build all the required images.
2. Then start them in a specific order.


## My Three Musketeers:
I have broken down my computer vision environment into three major docker images, I call them **The Three Musketeers**.
1. An image with deep learning and other programming packages installed - **The first Musketeer**.
2. An image with a Labeling/Annotation tool: [LabelMe](https://github.com/CSAILVision/LabelMeAnnotationTool) - **The second Musketeer**.
3. An image for database: [MongoDB](https://www.mongodb.com/) - **The third Musketeer**.


## Step 1: Building the Images:

### 1.1  The First Musketeer.
When it comes to choosing a deep learning library, I am a fan of [Tensorflow](https://www.tensorflow.org/). It is well supported and has the highest public community. The Main packages which we are going to install are:
* Python 3 - our programming language.
* Miniconda - our package manager.
* Tensorflow - our deep learning framework.
* Scikit learn, Numpy, Pandas etc - few useful Data science and ML Libraries.
* OpenCV - our image processing library.
* Keras - our interface for Tensorflow.
* Jupyter lab - our IDE. Accessed via browser.
* Pymongo - our interface for MongoDB.

First we need to configure the jupyter notebook with a default password ('deep_learning'). Feel free to change the password.
Create a file named **jupyter_notebook_config.py** and put the following lines inside it.

``` python
import os
from IPython.lib import passwd

c.NotebookApp.ip = '*'
c.NotebookApp.port = 8888
c.NotebookApp.open_browser = False

# sets a password if PASSWORD is set in the environment
c.NotebookApp.password = passwd('deep_learning')
```

Next let us define the Dockerfile which is used to build our **First Musketeer**:

**NOTE: The Dockerfile has the username `abhijit` change it to your username in the line `ENV NB_USER your_username` .**

``` dockerfile
ARG cuda_version=9.0
ARG cudnn_version=7
FROM nvidia/cuda:${cuda_version}-cudnn${cudnn_version}-devel

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
      bzip2 \
      g++ \
      git \
      graphviz \
      libgl1-mesa-glx \
      libhdf5-dev \
      openmpi-bin \
      wget && \
    rm -rf /var/lib/apt/lists/*

    # Install conda
ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

RUN wget --quiet --no-check-certificate https://repo.continuum.io/miniconda/Miniconda3-4.2.12-Linux-x86_64.sh && \
    echo "c59b3dd3cad550ac7596e0d599b91e75d88826db132e4146030ef471bb434e9a *Miniconda3-4.2.12-Linux-x86_64.sh" | sha256sum -c - && \
    /bin/bash /Miniconda3-4.2.12-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3-4.2.12-Linux-x86_64.sh && \
    echo export PATH=$CONDA_DIR/bin:'$PATH' > /etc/profile.d/conda.sh

# change username
ENV NB_USER abhijit

RUN useradd -m -s /bin/bash -N $NB_USER
RUN chown $NB_USER $CONDA_DIR -R
COPY jupyter_notebook_config.py /
RUN chown $NB_USER jupyter_notebook_config.py # thus the user can accerss the config file
RUN mkdir /abhijit_home
RUN chown $NB_USER -R /abhijit_home

USER $NB_USER


# Install Python packages and keras
ARG python_version=3.6

RUN conda install -y python=${python_version}
RUN pip install --upgrade pip
RUN pip install tensorflow-gpu==1.7.1
RUN conda install Pillow scikit-learn scikit-image matplotlib mkl nose pyyaml six h5py bokeh scipy numpy
RUN conda install pandas=0.20.2
RUN pip install keras==2.1.4
RUN pip install keras-vis==0.4.1
RUN pip install opencv-python==3.4.1.15

RUN conda install -c conda-forge jupyterlab \
                                 pymongo=3.6.1

RUN conda install -c anaconda protobuf=3.4.0

RUN conda clean -yt

#ENV PYTHONPATH='/src/:$PYTHONPATH'

# RUN mkdir /start_jupyter
# COPY run_jupyter.sh /start_jupyter
#COPY jupyter_notebook_config.py /root/.jupyter/
EXPOSE 8888 6006 27017

WORKDIR /abhijit_home
# CMD jupyter lab "$@" --no-browser --allow-root --port=8888 --ip=0.0.0.0
CMD jupyter lab --config=/jupyter_notebook_config.py --no-browser --port=8888 --ip=0.0.0.0
```
**Building the tensorflow image** :
Run the following command from the terminal:

``` bash
$docker build -t tensorflow_gpu_v1 .
```


### 1.2  The Second Musketeer:
Let us build a container containing a Labeling Tool. My choice is [LabelMe](https://github.com/CSAILVision/LabelMeAnnotationTool) because, it a web-based multi-user annotation tool. Out of the box it supports labeling for object classification, object detection and object instance segmentation. Thus making the perfect candidate for image labeling.

First we need to reconfigure apache for the tool to work in Ubuntu.

* Create a file named [000-default.conf](https://github.com/Abhijit-2592/dockerfiles/blob/master/labelme/000-default.conf) with the given contents in the link
* Create one more file named [apache2.conf](https://github.com/Abhijit-2592/dockerfiles/blob/master/labelme/apache2.conf) with the given contents in the link

Next let us define the Dockerfile which used to build our **Second Musketeer**:

**NOTE: The Dockerfile has the username `abhijit` change it to your username.**

``` dockerfile
FROM ubuntu:16.04

RUN useradd -s /bin/bash abhijit
# update ubuntu config
RUN apt-get update -y
RUN DEBIAN_FRONTEND=noninteractive apt-get -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" upgrade

# install dependencies for labelme
RUN apt-get install -y \
 				apache2 \
 				git \
 				libapache2-mod-perl2 \
 				libcgi-session-perl \
 				libapache2-mod-php \
 				make \
 				php

# Throws error
#RUN apt-get install php5 libapache2-mod-php5 -y

# Config apache
RUN a2enmod include
RUN a2enmod rewrite
RUN a2enmod cgi

# apache2 configuration: enabling SSI and perl/CGI scripts
COPY 000-default.conf /etc/apache2/sites-available/000-default.conf
COPY apache2.conf /etc/apache2/apache2.conf

#Clone LabelMe,move it and make
RUN git clone https://github.com/CSAILVision/LabelMeAnnotationTool.git
RUN mv ./LabelMeAnnotationTool/ /var/www/html/LabelMeAnnotationTool/
RUN cd /var/www/html/LabelMeAnnotationTool/ && make
RUN chown -R www-data:www-data /var/www/html

# change user
USER abhijit
# port binding
EXPOSE 80

# run
CMD ["/usr/sbin/apache2ctl", "-D", "FOREGROUND"]
```

**Building the labelme image** :
Run the following command from the terminal:

``` bash
$docker build -t labelme .
```


## 1.3 The Third Musketeer:
Finally lets build our third Musketeer which is a **MongoDB** database. Here instead of writing our own dockerfile we will pull the defaut image because, **why reinvent the wheel ?**

``` bash
$docker pull mongo:3.6.4
```


## Step 2: Starting the containers:
We will start the containers in the following order.

1. Mongodb container(our Third Musketeer) because we will link this container to the tensorflow container
2. LabelMe container(our Second Musketeer)
3. Tensorflow container(our First Musketeer)

Use the following bash scripts to start the containers


### 2.1 Starting the Third Musketeer:
``` bash
#!/bin/bash
# @Author: abhijit
# @Date:   24-May-2018 14-05-75
# @Last modified by:   abhijit
# @Last modified time: 24-May-2018 15-05-85
# pass the name
if ["$1" == ""]
    then
    echo "must be envoked only with the path to the directory which will act as storage for mongodb"
    exit 1
fi
mkdir -p $1
docker run \
--restart unless-stopped \
--name mongodb_database \
-p 27017:27017 \
-d \
-v $1:/data/db \
mongo:3.6.4
```


### 2.2 Starting the Second Musketeer:
``` bash
#!/bin/bash

echo "Starting labelme docker container!"
echo "Note:"
echo "*  This script will create the following directories $1/Images $1/Annotations $1/Masks $1/Scribbles $1/DirLists"
echo "*  The storage volumes are mounted outside the containers inside the parent directory $1 specified."
echo "*  This will delete the data in Images, Masks, Scribbles, Annotations, DirLists."
echo "*  Thus the default address will raise an error: ERROR IN FETCH IMAGE"
echo "*  To correct this error, manually add the example folders given in the official github repository to the folders created above!"
if ["$1" == ""]
    then
    echo "ERROR! No Argument Specified!"
    echo "must be envoked only with 1 argument: The path to the directory which will act as storage for LabelMe outside the container"
    exit 1
fi

mkdir -p $1/Images $1/Annotations $1/Masks $1/Scribbles $1/DirLists
docker run \
--name labelme \
-p 8080:80 \
-d \
-v $1/Images:/var/www/html/LabelMeAnnotationTool/Images \
-v $1/Annotations:/var/www/html/LabelMeAnnotationTool/Annotations \
-v $1/Masks:/var/www/html/LabelMeAnnotationTool/Masks \
-v $1/Scribbles:/var/www/html/LabelMeAnnotationTool/Scribbles \
-v $1/DirLists:/var/www/html/LabelMeAnnotationTool/annotationCache/DirLists \
--entrypoint "/bin/bash" \
-t labelme

# change ownership so that labelme can modify documents in mounted volumes
docker exec -u root labelme chown -R abhijit:www-data /var/www/html
docker exec -u root labelme chmod -R 774 /var/www/html

# restart apache inside the container
docker exec -u root labelme service apache2 restart
```

**NOTE**: You need to run `docker exec -u root labelme service apache2 restart` everytime after you restart your PC/Laptop.


### 2.3 Starting our First Musketeer:

``` bash
#!/bin/bash
# @Author: abhijit
# @Date:   24-May-2018 17-05-74
# @Last modified by:   abhijit
# @Last modified time: 24-May-2018 17-05-63

echo " "
echo "the following directories will be created and mounted to the container."
echo "base_path_defined_by_user/source_code as /source_code  where you can keep your source codes"
echo "base_path_defined_by_user/data  as /data    where you can keep the data used for deep_learning"
echo "base_path_defined_by_user/workspace as /workspace    workspace for you to work"
echo "base_path_defined_by_user/others as /others    for other requirements "
echo "  "
if ["$1" == ""]
    then
    echo "Usage Error!!!"
    echo "must be envoked only one argument: path to a base directory where all important datas are stored"
    exit 1
fi

# here connect to mongodb using data_mongo, 27017
mkdir -p $1/data $1/workspace $1/source_code $1/others
nvidia-docker run \
--name tf_keras \
--link mongodb_database:data_mongo \
-e DISPLAY=$DISPLAY \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-v $1:/abhijit_home \
-p 8888:8888 -p 6006:6006 \
tensorflow_gpu_v1
```


## Conclusion:
We have setup our deep learning for computer vision environment. If everything went well you should be able to access the following from your favourite browser:
* **Jupyter lab** from the url http://localhost:8888
* **LabelMe** from http://localhost:8080/LabelMeAnnotationTool/tool.html
* **MongoDB** is listening to port **27017** which can be accessed via the **Jupyter Lab**.

That's all folks!
