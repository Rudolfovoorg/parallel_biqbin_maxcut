### Connect to Trdina HPC
Open the terminal (Powershell on Windows) and enter:
```bash
ssh fisuserXXX@trdina-login.fis.unm.si
```
Enter your password when prompted.

### Download Biqbin solver
Clone the specific branch from github with the following command:
```bash
git clone --branch exascale_workshop --single-branch https://github.com/Rudolfovoorg/parallel_biqbin_maxcut.git
```
Go inside the folder:
```bash
cd parallel_biqbin_maxcut/
```
### Pull the workshop specific container
Biqbin needs a specific environment (libraries) to compile and run, for this we prepared an apptainer container for this workshop:
```bash
apptainer pull workshop-trdina.sif oras://docker.io/benozupanc/workshop-trdina:1.0.3
```

### Compile Biqbin
There is a Makefile available that will compile the source code into binaries: 
```bash
apptainer exec workshop-trdina.sif make
```

### Create and solve the QUBO from the stock data

Build a QUBO with this command, `selection_config.yaml` can be replaced with `quantities_config.yaml`.
```bash
apptainer exec workshop-trdina.sif python3 build_qubo.py selection_config.yaml
```

Run the solver on the HPC, pass in the specific qubo you want solved. `qubo_selection.json` can be replaced with `qubo_quantities.json`
```bash
 sbatch run.sh qubo_selection.json
 ```

Decode the QUBO solution back into stock portfolio information:
```bash
apptainer exec workshop-trdina.sif python3 decode_solution.py qubo_selection.json
```