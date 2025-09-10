FROM python:3.12 

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    make \
    liblapack-dev \
    liblapack3 \
    libopenblas-dev \
    mpich \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install numpy


WORKDIR /solver
COPY . .

RUN pip install -r requirements.txt 

ENV OPENBLAS_NUM_THREADS=1 
ENV GOTO_NUM_THREADS=1 
ENV OMP_NUM_THREADS=1

RUN make
RUN make test