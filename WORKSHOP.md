```bash
git clone 
```

```bash
apptainer pull workshop-trdina.sif oras://docker.io/benozupanc/workshop-trdina:1.0.2
```
```bash
apptainer exec workshop-trdina.sif make
```

```bash
apptainer exec workshop-trdina.sif python3 build_qubo.py selection_config.yaml
```

```bash
apptainer exec workshop-trdina.sif mpirun python3 biqbin_qubo.py qubo_name
```

```bash
apptainer exec workshop-trdina.sif python3 decode_solution.py qubo_name
```