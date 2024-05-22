from datetime import datetime
import subprocess
import sys

def get_optimal_qos():
    output = subprocess.getoutput('squeue')
    curr_jobs = output.count('arvind')
    if curr_jobs < 2:
        return 'high'
    elif curr_jobs < 3:
        return 'medium'
    elif curr_jobs < 5:
        return 'default'
    else:
        return 'scavenger'

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: {} <command>'.format(sys.argv[0]))
        sys.exit(1)

    command = ' '.join(sys.argv[1:])
    timestamp = int(datetime.now().timestamp())

    JOB_NAME = 'crafter_dqn'
    TIME = '12:00:00'
    GPUS = 1
    GPU_SHARDS = 4
    CPUS_PER_TASK = 1
    MEM = '8gb'

    args_dict = {
        'job-name': JOB_NAME,
        'time': TIME,
        'gpus': GPUS,
        'cpus-per-task': CPUS_PER_TASK,
        'mem': MEM,
        'qos': get_optimal_qos(),
        'output': f'slurm/{timestamp}-out.txt',
        'error': f'slurm/{timestamp}-err.txt'
    }

    total_memory = int(subprocess.getoutput('nvidia-smi -i 0 --format=csv,noheader,nounits --query-gpu=memory.total'))
    target_memory = round(GPU_SHARDS * 1024 * 0.9)
    fraction = target_memory / total_memory

    # Write to a file
    with open(f'slurm/command.sh'.format(timestamp), 'w') as f:
        f.write('#!/bin/bash\n')
        for k in args_dict:
            f.write(f'#SBATCH --{k}={args_dict[k]}\n')
        f.write('\n')
        f.write(f'TOTAL_MEMORY={total_memory}\n')
        f.write(f'TARGET_MEMORY={target_memory}\n')
        f.write(f'FRACTION={fraction}\n')
        f.write('echo "Allocating $FRACTION of GPU ($TARGET_MEMORY MiB out of $TOTAL_MEMORY MiB)"\n')
        f.write('export XLA_PYTHON_CLIENT_MEM_FRACTION=$FRACTION\n')
        f.write('\n')
        f.write('srun ' + 'conda run --no-capture-output -n diayn4 ' + command + '\n')

    print(f'Logging at: slurm/{timestamp}-out.txt and slurm/{timestamp}-err.txt')
    print('Running...')

    subprocess.run(f'chmod u+x slurm/command.sh'.split(' '))
    script = subprocess.Popen(f"sbatch slurm/command.sh".split(' '), stdin=subprocess.PIPE)
    return_code = script.wait()
    print('Done. Return code:', return_code)

# EXAMPLE: python slurm.py diayn.py -c lunar_raw_s3.yml
