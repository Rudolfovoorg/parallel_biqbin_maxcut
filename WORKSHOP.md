```bash
git clone 
```
```bash
apptainer exec biqbin-trdina.sif make
```

```bash
apptainer exec biqbin-trdina.sif python3 build_qubo.py selection_config.yaml
```

```bash
apptainer exec biqbin-trdina.sif mpirun python3 biqbin_qubo.py qubo_name
```

```bash
apptainer exec biqbin-trdina.sif python3 decode_solution.py qubo_name
```