#!/bin/bash
#SBATCH --ntasks=1000
#SBATCH --time=48:00:00
#SBATCH --partition=cpu

export UCX_TLS=self,sm,rc,ud
export OMPI_MCA_PML="ucx"
export OMPI_MCA_osc="ucx"

export OPENBLAS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export OMP_NUM_THREADS=1

srun python3 biqbin_plzen.py $1