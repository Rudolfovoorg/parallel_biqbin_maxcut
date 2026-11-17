FROM python:3.12-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    make \
    libopenblas-dev \
    mpich \
    libmpich-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /solver
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV OPENBLAS_NUM_THREADS=1 \
    GOTO_NUM_THREADS=1 \
    OMP_NUM_THREADS=1

COPY . .
RUN make
