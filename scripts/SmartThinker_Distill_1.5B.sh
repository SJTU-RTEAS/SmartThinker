#!/bin/bash
set -x

# Configuration through environment variables
# Set these variables before running:
cur_dir=$(dirname -- "$(readlink -f -- "$0")")
root_dir=$(dirname "$cur_dir")
if [ -f .env ]; then
  set -a
  source ${root_dir}.env
  set +a
fi
export PROJECT_HOME=${root_dir}
export LOG_DIR="${PROJECT_HOME}/log"
export DATASET_DIR=${PROJECT_HOME}/datasets

export PYTHONPATH="${PROJECT_HOME}:$PYTHONPATH"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

# Default model path if not specified
MODEL_PATH=${MODEL_PATH:-"~/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"}

use_dynamic_bsz=True

# Train over a single node, 4 PRO6000-96GB GPUs.
python3 ${PROJECT_HOME}/src/trainer/main_SmartThinker.py \
    algorithm.adv_estimator=grpo_smartthinker \
    data.train_files=${DATASET_DIR}/deepscaler_preview.parquet \
    data.val_files=${DATASET_DIR}/aime25.parquet \
    data.train_batch_size=64 \
    data.max_prompt_length=1500 \
    data.max_response_length=8000 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.offload_policy=False \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='SmartThinker' \
    trainer.experiment_name='Distill-1.5B' \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.test_freq=-1 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=3 \
    custom_reward_function.path=${PROJECT_HOME}/src/utils/rewards/math_compute_score.py \
    custom_reward_function.name=math_compute_score \
    reward_manager.source=importlib \
    reward_manager.name=SmartThinkerRewardManager \
    reward_manager.module.path=${PROJECT_HOME}/src/workers/reward_manager/SmartThinker.py \
    reward_manager.module.name=SmartThinkerRewardManager \
    reward_model.enable=False \
    reward_model.use_reward_loop=False \
    reward_model.nnodes=1 \
    "${@:1}" 