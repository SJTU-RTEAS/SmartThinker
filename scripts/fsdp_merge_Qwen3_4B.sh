#!/bin/bash

project_name="ShorterBetter"
exp_name="Qwen3-4B"
step=50

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir checkpoints/$project_name/$exp_name/global_step_$step/actor/ \
    --target_dir checkpoints/$project_name/$exp_name/global_step_$step/actor/huggingface