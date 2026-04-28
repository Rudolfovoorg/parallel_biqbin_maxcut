#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --partition=cpu
#SBATCH --mem=132GB

# export environment variables
export UCX_TLS=self,sm,rc,ud
export OMPI_MCA_PML="ucx"
export OMPI_MCA_osc="ucx"

export OPENBLAS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Run the solver
srun python3 biqbin_qubo.py $1 -c --output $2