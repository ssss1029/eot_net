# -*- coding: utf-8 -*-

"""
Given a bunch of commands to run, check the available GPUs and run them on the GPUs in separate tmux sessions.
Usage: Just modify the settings in the Config class and then run python3 gpu_run.py
"""

import GPUtil
import subprocess
import sys
import time

class Config:
    """
    Global class that houses all configurations
    """
    
    # Shared args to put onto all of the JOBS
    SHARED_ARGS = ""

    # Specifies tasks to run. It maps tmux session name to the command to run in that session.
    JOBS = {
        "baseline_ImageNetR_classes": {
            "num_gpus": 2,
            "command": "python3 train_imagenet.py --data-standard=/var/tmp/namespace/hendrycks/imagenet/train --data-val=/var/tmp/namespace/hendrycks/imagenet/val --save=checkpoints/baseline_ImageNetR_classes --arch=resnet18 --lr=0.1 --epochs=100 --workers=20 --batch-size=512"
        }
    }

    # Time to wait between putting jobs on GPUs (in seconds). This is useful because it might take time 
    # for a process to actually load the network onto the GPU, so we wait until that is done before 
    # selecting the GPU for the next process.
    SLEEP_TIME = 10

    # Minimum memory required on a GPU to consider putting a job on it (MiB).
    MIN_MEMORY_REQUIRED = 2000


# Stick the shared args onto each JOB
for key, job in Config.JOBS.items():
    new_command = job['command'] + " " + Config.SHARED_ARGS
    Config.JOBS[key]['command'] = new_command

def select_gpu(GPUs, number):
    """
    Select the next best available GPU to run on. If nothing exists, return None
    """
    GPUs = list(filter(lambda gpu: gpu.memoryFree > Config.MIN_MEMORY_REQUIRED, GPUs))
    if len(GPUs) == 0:
        return None
    GPUs = sorted(GPUs, key=lambda gpu: gpu.memoryFree)
    return GPUs[-(number):]

for index, (tmux_session_name, job) in enumerate(Config.JOBS.items()):
    command = job['command']
    num_gpus = job['num_gpus']

    # Get the best available GPU
    print("Finding GPU for command \"{0}\"".format(command))
    curr_gpus = select_gpu(GPUtil.getGPUs(), num_gpus)

    if curr_gpus == None or len(curr_gpus) != num_gpus:
        print("No available GPUs found. Exiting.")
        sys.exit(1)

    print("SUCCESS! Found GPU ids = {0} which have {1} MiB free memory".format(
        [c.id for c in curr_gpus], [c.memoryFree for c in curr_gpus]
    ))

    gpu_ids_string = ",".join([str(c.id) for c in curr_gpus])

    result = subprocess.run("tmux new-session -d -s {0}".format(tmux_session_name), shell=True)        
    if result.returncode != 0:
        print("Failed to create new tmux session called {0}".format(tmux_session_name))
        sys.exit(result.returncode)

    result = subprocess.run("tmux send-keys 'CUDA_VISIBLE_DEVICES={0} {1}' C-m".format(
        gpu_ids_string, command
    ), shell=True)
    if result.returncode != 0:
        print("Failed to run {0} in tmux session".format(command, tmux_session_name))
        sys.exit(result.returncode)

    print("---------------------------------------------------------------")

    if index < len(Config.JOBS) - 1:
        time.sleep(Config.SLEEP_TIME)
