#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --partition=cpu

# export environment variables
export UCX_TLS=self,sm,rc,ud
export OMPI_MCA_PML="ucx"
export OMPI_MCA_osc="ucx"

export OPENBLAS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Run the solver
srun --mpi=pmix python3 biqbin_qubo.py $1 --output solutions/$1-$2.json -c -t 2760