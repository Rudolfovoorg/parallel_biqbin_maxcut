
# ============================================================
# BiqBin - Multi-stage Dockerfile
#
# Targets:
#   runtime (default): Solver and Python bindings
#   ci:                Runtime + development/test dependencies
# ============================================================

ARG PYTHON_VERSION=3.12

# ============================================================
# Base - System and Python runtime dependencies
# ============================================================

FROM python:${PYTHON_VERSION}-bookworm AS base

ARG MPI_IMPL=mpich

RUN set -eux; \
    apt-get update; \
    case "$MPI_IMPL" in \
      mpich) \
        apt-get install -y --no-install-recommends \
          build-essential \
          make \
          libopenblas-dev \
          mpich \
          libmpich-dev ;; \
      openmpi) \
        apt-get install -y --no-install-recommends \
          build-essential \
          make \
          libopenblas-dev \
          openmpi-bin \
          libopenmpi-dev ;; \
      *) \
        echo "Unsupported MPI implementation: $MPI_IMPL" >&2; \
        exit 1 ;; \
    esac; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /solver

# Install Python runtime dependencies separately to
# preserve Docker's dependency cache.
COPY requirements.txt ./

RUN python -m pip install --no-cache-dir \
    -r requirements.txt

# Prevent thread oversubscription across MPI ranks.
ENV OPENBLAS_NUM_THREADS=1 \
    GOTO_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1


# ============================================================
# Build - Compile BiqBin
# ============================================================

FROM base AS build

ARG LOG_LEVEL=INFO

COPY . .

# Compile both the native executable and Python bindings.
RUN make LOG_LEVEL="${LOG_LEVEL}"


# ============================================================
# CI - Build + test/development dependencies
# ============================================================

FROM build AS ci

RUN python -m pip install --no-cache-dir \
    -r requirements-dev.txt

# Allow OpenMPI test execution inside Docker.
#
# Root execution is permitted only in the CI image.
# Oversubscription allows MPI tests to launch more ranks
# than Docker reports as available CPU slots.
ENV OMPI_ALLOW_RUN_AS_ROOT=1 \
    OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    OMPI_MCA_rmaps_base_oversubscribe=1

CMD ["/bin/bash"]


# ============================================================
# Runtime - Default image for users
# ============================================================

FROM build AS runtime

CMD ["/bin/bash"]
