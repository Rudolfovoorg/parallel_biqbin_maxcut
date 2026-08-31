#!/bin/bash
#SBATCH --ntasks-per-node=60
#SBATCH --nodes=1
#SBATCH --time=24:00:00

# export environment variables
export OPENBLAS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Run the solver
apptainer exec biqbin.sif mpirun python3 biqbin_run_all_bqp.py $1