## Example of how to setup and run Biqbin on an HPC environment
The following example will be for [HPC Vega - Izum](https://en-vegadocs.vega.izum.si/), your HPC environment may vary.

### Installation

Clone the project in the desired directory:

```bash
git clone https://github.com/Rudolfovoorg/parallel_biqbin_maxcut.git
```
Enter into the directory:

```bash
cd parallel_biqbin_maxcut/
```
Load modules required by Biqbin:

```bash
module load OpenMPI/4.1.6-GCC-13.2.0
module load OpenBLAS/0.3.24-GCC-13.2.0
module load Python/3.11.5-GCCcore-13.2.0
```

Install the requirements:

```bash
pip install -r requirements.txt
```

Compile:

```bash
make
```

After this you will have a working version of Biqbin setup.

### Running the program

Remember to load in the three modules on every new session:


```bash
module load OpenMPI/4.1.6-GCC-13.2.0
module load OpenBLAS/0.3.24-GCC-13.2.0
module load Python/3.11.5-GCCcore-13.2.0
```

Create a bash script for your job called `run.sh`:

```bash
#!/bin/bash
#SBATCH --ntasks=10
#SBATCH --nodes=2
#SBATCH --time=02:00:00
#SBATCH --partition=cpu
#SBATCH --output=output_file_path
#SBATCH --job-name=example_job

# export environment variables
export UCX_TLS=self,sm,rc,ud
export OMPI_MCA_PML="ucx"
export OMPI_MCA_osc="ucx"

export OPENBLAS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Run the solver
mpirun python3 biqbin_qubo.py tests/qubos/40/kcluster40_025_10_1.json
```

The above example runs the QUBO version of the solver on a test instance, for other examples please refer to the [readme file](README.md).

For `#SBATCH` variables please refer to your [HPC documentation](https://en-vegadocs.vega.izum.si/first-job/).

Submit the job:

```bash 
sbatch run.sh
```

