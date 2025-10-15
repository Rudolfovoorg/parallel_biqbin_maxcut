#!/bin/bash
#SBATCH --ntasks=10
#SBATCH --nodes=1
#SBATCH --time=00:01:00
#SBATCH --output=stdout_example_job
#SBATCH --job-name=example_job

# export environment variables
export OPENBLAS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Run the solver
apptainer exec workshop-trdina.sif mpirun python3 biqbin_qubo.py "$1"