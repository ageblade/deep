**TensorFlow** is an open source software library for numerical computation using
data flow graphs.  Nodes in the graph represent mathematical operations, while
the graph edges represent the multidimensional data arrays (tensors) that flow
between them.  This flexible architecture lets you deploy computation to one
or more CPUs or GPUs in a desktop, server, or mobile device without rewriting
code.  TensorFlow also includes TensorBoard, a data visualization toolkit.

# Using TensorFlow via Docker

This directory contains `Dockerfile`s to make it easy to get up and running with
TensorFlow via [Docker](http://www.docker.com/) with the addition of image processing utils 
such as GDAL.

## Running the container

Run non-GPU container using

    $ docker run --name=tensorflow -itd -p 8888:8888 ageblade/tensorflow:latest
	$ docker exec -it tensorflow /bin/bash




