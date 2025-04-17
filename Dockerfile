FROM ubuntu:22.04 

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    mpich \
    python3 \
    python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir scipy pytest

# Set environment variables to avoid multithreaded math lib conflicts
ENV OPENBLAS_NUM_THREADS=1
ENV GOTO_NUM_THREADS=1
ENV OMP_NUM_THREADS=1

WORKDIR /solver

RUN chmod -R 777 .

COPY . .

USER 1001:1001

RUN make clean
RUN make
RUN make test
RUN pytest -v
