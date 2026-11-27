# Installation

BiqBin supports native Linux builds and Docker-based environments.

For HPC installations using environment modules and a batch scheduler, see the [HPC guide](HPC_EXAMPLE.md).

## Requirements

A native build requires:

- Linux
- Python 3.12 or later
- GCC and G++
- GNU Make
- Python development headers
- MPICH or OpenMPI
- OpenBLAS
- Python packages from `requirements.txt`

The build uses `mpicc` and `mpic++`, so both compiler wrappers must be available in your `PATH`.

## Native Linux installation

The commands below target Ubuntu 24.04 with Python 3.12. On other Linux
distributions, use equivalent packages and Python 3.12 or later, or follow
the Conda or Docker instructions below. CI tests Python 3.12 and 3.13.

Run project commands from the repository root. Install the system dependencies:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  python3-dev \
  python3-venv \
  mpich \
  libmpich-dev \
  libopenblas-dev
```

You may use OpenMPI instead of MPICH:

```bash
sudo apt-get install -y \
  build-essential \
  python3-dev \
  python3-venv \
  openmpi-bin \
  libopenmpi-dev \
  libopenblas-dev
```

Use one MPI implementation consistently when compiling and running the solver.

Create and activate a virtual environment, then install the runtime requirements.
The environment provides its own pip; do not install into the system Python.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Activate this environment again with `source .venv/bin/activate` in each new
shell before building or running BiqBin.

Build the project:

```bash
make
```

For a complete rebuild:

```bash
make rebuild
```

The default `make` command uses incremental compilation. A full rebuild is recommended when changing compilers, MPI implementations or compilation options.

## Conda installation

BiqBin can also be built inside a Conda environment.

### 1. Install Conda

Install [Miniconda](https://docs.anaconda.com/miniconda/) or another compatible Conda distribution.

Follow the official installation instructions for your Linux distribution.

### 2. Create an environment

```bash
conda create -n biqbin-py312 python=3.12
conda activate biqbin-py312
```

### 3. Install native dependencies

For an OpenMPI environment:

```bash
conda install -c conda-forge \
  compilers \
  make \
  openmpi \
  openblas
```

Alternatively, install MPICH from conda-forge rather than OpenMPI.

The MPI compiler wrappers used for building must correspond to the MPI runtime used for execution.

Verify:

```bash
which mpicc
which mpic++
mpicc -show
```

### 4. Install Python requirements

For normal solver execution:

```bash
python -m pip install -r requirements.txt
```

For development and testing:

```bash
python -m pip install -r requirements-dev.txt
```

### 5. Build

```bash
make
```

Verify the Python package imports successfully:

```bash
python -c "import biqbin; print('BiqBin import successful')"
```

When switching between Conda environments or MPI implementations, clean the previous native objects before rebuilding:

```bash
make clean
make
```

## Docker

The root [Dockerfile](https://github.com/Rudolfovoorg/parallel_biqbin_maxcut/blob/main/Dockerfile) defines a shared native build environment and two selectable targets.

| Target | Description |
|---|---|
| `runtime` | Default compiled solver environment |
| `ci` | Compiled solver with development and testing dependencies |

Docker uses Python 3.12, Debian Bookworm and MPICH by default.

The runtime image runs as the nonroot user `biqbin`, with UID and GID `1000` by
default. Its working directory is `/solver`. The bundled source and examples,
`/data`, and the user's home directory `/home/biqbin` are writable by that user.

### Build the runtime image

```bash
docker build -t parallel-biqbin:1.0.0 .
```

Equivalent Makefile target:

```bash
make docker
```

Open a shell:

```bash
docker run --rm -it parallel-biqbin:1.0.0 bash
```

Or use:

```bash
make docker-shell
```

The Makefile command mounts the configured host data directory into `/data`.

### Build the development image

```bash
docker build --target ci -t parallel-biqbin-dev:1.0.0 .
```

Or:

```bash
make docker-dev
```

The CI image additionally installs `requirements-dev.txt`, including pytest, Ruff and coverage tooling.

Run the complete suite:

```bash
docker run --rm --init parallel-biqbin-dev:1.0.0 make test
```

Alternatively:

```bash
make docker-test
```

### Select a Python version

The Dockerfile accepts `PYTHON_VERSION` as a build argument:

```bash
docker build \
  --target runtime \
  --build-arg PYTHON_VERSION=3.13 \
  -t biqbin-py313 .
```

### Select an MPI implementation

The `MPI_IMPL` build argument supports `mpich` and `openmpi`.

For example:

```bash
docker build \
  --target runtime \
  --build-arg MPI_IMPL=openmpi \
  -t biqbin-openmpi .
```

### Compile-time logging configuration

The default native logging threshold is INFO.

To build with DEBUG diagnostics compiled in:

```bash
docker build \
  --target runtime \
  --build-arg LOG_LEVEL=DEBUG \
  -t biqbin-debug .
```

This build argument determines the compiled native log threshold, independently of runtime CLI verbosity.

### Runtime user and host permissions

The `BIQBIN_UID` and `BIQBIN_GID` build arguments select the runtime account's
numeric user and group IDs. Both default to `1000` and must be greater than zero.
The UID must be unused in the base image; an existing group with the requested
GID is reused.

On Linux, bind-mounted directories retain their host ownership and permissions.
The image's ownership settings for `/data` do not change a directory mounted
over it. To build the default runtime image with your own IDs, run this from your
regular Linux account:

```bash
docker build \
  --target runtime \
  --build-arg BIQBIN_UID="$(id -u)" \
  --build-arg BIQBIN_GID="$(id -g)" \
  -t parallel-biqbin:1.0.0 .
```

These arguments can also be combined with `MPI_IMPL`, `PYTHON_VERSION`, and
`LOG_LEVEL`. Matching IDs allows the container user to write files that your
host account can write, subject to the mount's permissions.

### Running with local input files

Mount a directory containing problem instances:

```bash
mkdir -p instances
docker run --rm -it \
  -v "$PWD/instances:/data" \
  parallel-biqbin:1.0.0 bash
```

Inside the container, the host files are available under `/data`.

For a Max-Cut JSON file named `instance.json` in that directory, write the result
and per-rank logs back to the mounted directory:

```bash
mpiexec -n 3 python biqbin_maxcut.py \
  /data/instance.json \
  -o /data/instance.result.json \
  --log-dir /data/instance.logs
```

The native executable writes its output beside the input file, so the input
directory must be writable. Files written only inside the container, outside
the mounted directory, are removed when the container is deleted with `--rm`.

### OpenMPI in containers

The `ci` image permits root-based OpenMPI execution and enables oversubscription so small MPI tests can run on CPU-constrained CI environments.

The regular runtime target uses the nonroot `biqbin` user, so OpenMPI can launch
without root-execution overrides. Oversubscription is not enabled by default.

For a small local test on a machine with fewer than three available MPI slots,
add `--oversubscribe` to the `mpiexec` command. This changes CPU-slot allocation;
it does not change the container user.

For production or benchmarking runs, use an appropriate MPI process count and CPU allocation.

## Testing and verification

The following commands require the development dependencies:

```bash
python -m pip install -r requirements-dev.txt
```

Run the full regression suite:

```bash
make test
```

Run the smaller unit-test subset:

```bash
make test-unit
```

Collect Python coverage:

```bash
make coverage
```

Check Python linting and formatting:

```bash
make lint
make format-check
```

Coverage output is written to `artifacts/coverage/.coverage`, `artifacts/coverage/coverage.xml` and `artifacts/coverage/htmlcov/`. JUnit XML reports are produced in `artifacts/tests/junit/`.

Some MPI integration tests generate output files used by later validation tests. Run the complete suite through the Makefile rather than invoking all pytest files indiscriminately.

## Documentation

Install the documentation dependencies:

```bash
python -m pip install -r requirements-docs.txt
```

Build the documentation:

```bash
make docs
```

Preview locally:

```bash
make docs-serve
```

The generated site is stored in `site/`.

Documentation generation does not require compiling the solver or importing the native MPI extension.

## Troubleshooting

### MPI compiler not found

Check:

```bash
which mpicc
which mpic++
```

Make sure the intended MPI development package is installed and its compiler wrappers are available.

### Python extension import fails

Verify that you are using the same Python environment used to compile the native extension.

If necessary:

```bash
make clean
make
```

### OpenMPI reports insufficient slots

OpenMPI may detect fewer available processor slots than the requested MPI rank count.

For local correctness testing, allocate sufficient CPU resources or explicitly enable oversubscription. Avoid oversubscription for performance measurements.

### OpenBLAS uses too many threads

Prevent BLAS thread oversubscription, particularly when running multiple MPI ranks:

```bash
export OPENBLAS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export OMP_NUM_THREADS=1
```

### Tests fail after switching build options

GNU Make does not automatically detect compiler or build-variable changes.

Clean and rebuild when switching MPI implementations, Python environments or native logging thresholds:

```bash
make clean
make
```

For further build and test commands, see the repository [Makefile](https://github.com/Rudolfovoorg/parallel_biqbin_maxcut/blob/main/Makefile).