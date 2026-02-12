#!/bin/bash

project_name="ShorterBetter"
exp_name="Distill-7B"
step=75

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir checkpoints/$project_name/$exp_name/global_step_$step/actor/ \
    --target_dir checkpoints/$project_name/$exp_name/global_step_$step/actor/huggingface