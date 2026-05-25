
text = ''
for problem in ['problem_instance_40.json', 'problem_instance_160.json']:
    for num_cores in [2**i for i in range(2, 12)]:
        print(num_cores)
        text += f'SBATCH --ntasks={num_cores} --job-name={problem}-{num_cores} --output={problem}-{num_cores}-stdout job_srun.sh {problem} {num_cores}\n'
    with open(f'send_{problem}.sh', 'w') as f:
        f.write(text)
    text = ''