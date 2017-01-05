Builds:

- [TensorFlow](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/docker)
- GDAL ([gdal-docker] (https://github.com/geo-data/gdal-docker))

## Running the container

Run non-GPU container using

    $ docker run --name=deep -itd -p 8888:8888 ageblade/deep:latest
	$ docker exec -it deep /bin/bash




