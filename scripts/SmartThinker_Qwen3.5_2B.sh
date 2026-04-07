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
#export VLLM_ATTENTION_BACKEND=FLASH_ATTN

#export CUDA_LAUNCH_BLOCKING=1
#export HYDRA_FULL_ERROR=1
# export NCCL_P2P_DISABLE=1

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
MODEL_PATH=${MODEL_PATH:-"/home/huchenzhi/models/Qwen/Qwen3.5-2B"}

use_dynamic_bsz=False
sp_size=1 

rollout_is=sequence
rollout_is_threshold=2.0
rollout_is_batch_normalize=true
rollout_rs=token_k1
rollout_rs_threshold=0.6_1.6

# Train over a single node, 4 PRO6000-96GB GPUs.
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 ${PROJECT_HOME}/src/trainer/main_SmartThinker.py \
    algorithm.adv_estimator=grpo_smartthinker \
    critic.enable=false \
    data.train_files=${DATASET_DIR}/deepvision-math.parquet \
    data.val_files=${DATASET_DIR}/mathvision-mini.parquet \
    data.train_batch_size=64 \
    data.max_prompt_length=2048 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    data.shuffle=False \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=1 \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True \
    actor_rollout_ref.ref.fsdp_config.reshard_after_forward=True \
    actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=True \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.actor.fsdp_config.offload_policy=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=$sp_size \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.ref.fsdp_config.offload_policy=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.ignore_eos=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=10240 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16384 \
    algorithm.use_kl_in_reward=False \
    trainer.use_legacy_worker_impl=auto \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='SmartThinker' \
    trainer.experiment_name='Qwen3.5-2B' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    reward.custom_reward_function.path=${PROJECT_HOME}/src/utils/rewards/math_compute_score.py \
    reward.custom_reward_function.name=math_compute_score \
    reward.reward_manager.source=importlib \
    reward.reward_manager.name=SmartThinkerRewardManager \
    reward.reward_manager.module.path=${PROJECT_HOME}/src/workers/reward_manager/SmartThinker_async.py \
    reward.reward_manager.module.name=SmartThinkerRewardManager \
    reward_model.enable=False \
    reward_model.nnodes=1 \
    trainer.balance_batch=False \
    trainer.val_before_train=False \
    trainer.save_freq=25 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_cascade_attn=True \
    "${@:1}" 