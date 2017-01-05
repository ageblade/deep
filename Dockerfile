FROM tensorflow/tensorflow:latest

RUN apt-get update -y && apt-get install -y \
	git \
	make

ENV GDAL_DOCKER_REPO https://github.com/geo-data/gdal-docker.git
ENV GDAL_DOCKER_BRANCH 2.1.2
ENV GDAL_DOCKER_PATH /gdal-docker

# Install GDAL using the gdal-docker makefile
RUN set -x \
	&& cd / && git clone $GDAL_DOCKER_REPO $GDAL_DOCKER_PATH \
	&& cd $GDAL_DOCKER_PATH && git checkout $GDAL_DOCKER_BRANCH \
	&& make install clean \ 
	&& rm -rf $GDAL_DOCKER_PATH

# Install python dependencies
RUN apt-get install -y \ 
	libspatialindex-dev \
	&& pip --no-cache-dir install \
    pandas \
	rtree \
	shapely

# Remove additional installed dependencies
RUN apt-get purge -y \
	make
	
ENV SRC_PATH /sources
	
ADD . $SRC_PATH