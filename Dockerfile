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

WORKDIR /boost

RUN wget https://archives.boost.io/release/1.88.0/source/boost_1_88_0.tar.gz
RUN tar xzvf boost_1_88_0.tar.gz

WORKDIR /boost/boost_1_88_0

RUN ./bootstrap.sh --prefix=/usr/
RUN ./b2
RUN ./b2 install

WORKDIR /solver
COPY . . 

RUN pip install -r requirements.txt 

ENV PYBOOST_INCLUDES="-I/usr/local/include/python3.12 -I/usr/include"
ENV PYBOOST_LIBS="-L/usr/lib -lboost_python312 -lpython3.12 -lboost_numpy312"
ENV OPENBLAS_NUM_THREADS=1 
ENV GOTO_NUM_THREADS=1 
ENV OMP_NUM_THREADS=1

RUN make clean
RUN make
RUN make test