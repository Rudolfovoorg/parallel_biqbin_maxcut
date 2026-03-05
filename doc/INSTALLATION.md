# Installation guide for Conda and Docker

### Conda based build

This project is fully buildable inside a Conda environment.

### Setup Instructions (Conda Environment)

#### 1. Install Anaconda (if not already)

Download

```bash
curl -O https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
```

Install

```bash
bash ~/Anaconda3-2024.10-1-Linux-x86_64.sh
```

---

Configure solver if not set to libmamba

```bash
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```

Update to newest

```bash
conda update -n base -c defaults conda
```

#### 2. Create and activate Conda environment

```bash
conda create -n biqbin-py312 python=3.12
```

```bash
conda activate biqbin-py312
```

#### 3. Install dependencies

C and C++ compiler

```bash
conda install conda-forge::gxx -y
```

MPI

```bash
conda install conda-forge::openmpi -y
```

OpenBLAS

```bash
conda install conda-forge::openblas -y
```

Python packages:

```bash
pip install -r requirements.txt
```
or
```bash
pip install -r requirements-dev.txt
```

#### 5. Compile using Makefile

```bash
make
```

## Docker

[**`Dockerfile`**](../Dockerfile) available for running the solver in a docker container, image can be created using Makefile command

```bash
make docker
```

To access the docker container use:

```bash
make docker-shell
```

for other specific commands please check the [`Makefile`](../Makefile)
