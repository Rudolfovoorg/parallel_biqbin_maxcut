FROM ubuntu:22.04 

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    make \
    liblapack-dev \
    liblapack3 \
    libopenblas-dev \
    mpich \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /solver

RUN chmod -R 777 .

COPY . .

USER 1001:1001

RUN make clean
RUN make