set -euo pipefail

cd verl/

mix_rollout=True
mix_probs=True
mix_ratio=0.875
ref_n=4
decay_steps=0

reload_for_mix=False
reload_for_mix_path=None

replace_old_probs=False
replace_on_ref_only=True
replace_method="mix"
renorm_adv="none"
quantile_k=-1

order_by="none"

ref_policy_sync_steps=1
project_name='ARMOR'
exp_name="Qwen3-8B-Base-restart_DAPO-mix_anchored_${mix_ratio}-sync_${ref_policy_sync_steps}-only_cor_ref_${ref_n}"

use_ref_model=True
# NOTE: (1st run, DAPO) In our experiments, we first train DAPO on Qwen3-8B-Base model, pick the best-performing ckpt before over-optimization,
# (2nd run, ARMOR) and put the ckpt here for continual training
ref_model=${MODEL_TO_CONSUME:-"Put the model ckpt here for continual training with ARMOR"}
MODEL_PATH=${ref_model}

nccl_timeout=36000

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 20))
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=10
train_prompt_bsz=512
gen_prompt_bsz=$((train_prompt_bsz * 3))
n_resp_per_prompt=16
train_prompt_mini_bsz=32

WORKING_DIR=${WORKING_DIR:-"${PWD}"}
echo "Working directory: ${WORKING_DIR}"
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
NNODES=${NNODES:-16}

# paths
RAY_DATA_HOME=${RAY_DATA_HOME:-"/mgfs/shared/Group_GY/zhenqiu.hkx/codes/myrl/verl_home"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/dapo-math-17k.parquet"}

aime24_file="/mgfs/shared/Group_GY/chiyum/data/aime2024/aime2024.parquet"
aime25_file="/mgfs/shared/Group_GY/chiyum/data/aime25_dapo/aime2025_dapo.parquet"
TEST_FILE="['$aime24_file','$aime25_file']"

# Rollout and validation data 
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-"${RAY_DATA_HOME}/rollout_data/${project_name}/${exp_name}"}
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-"${RAY_DATA_HOME}/validation_data/${project_name}/${exp_name}"}

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7

# Performance Related Parameter
sp_size=4
use_dynamic_bsz=True
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
offload=True
gen_tp=2

python3 -m recipe.dapo.main_dapo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    +algorithm.use_ref_model=${use_ref_model} \
    +algorithm.mix_rollout.ref_n=${ref_n} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.mix_rollout.enable=${mix_rollout} \
    algorithm.mix_rollout.main_ratio=0.9375 \
    +algorithm.mix_rollout.auto_mix_ratio=False \
    algorithm.mix_rollout.decay_steps=${decay_steps} \
    algorithm.mix_rollout.reload_for_mix=${reload_for_mix} \
    algorithm.mix_rollout.reload_for_mix_path=${reload_for_mix_path} \
    +algorithm.quantile_k=${quantile_k} \
    +algorithm.ref_policy_sync_steps=${ref_policy_sync_steps} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.checkpoint.save_contents='["model", "optimizer", "extra", "hf_model"]' \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    +actor_rollout_ref.actor.mix_probs=${mix_probs} \
    +actor_rollout_ref.actor.mix_probs_main_ratio=${mix_ratio} \
    +actor_rollout_ref.actor.replace_old_probs=${replace_old_probs} \
    +actor_rollout_ref.actor.replace_on_ref_only=${replace_on_ref_only} \
    +actor_rollout_ref.actor.replace_method=${replace_method} \
    +actor_rollout_ref.actor.renorm_adv=${renorm_adv} \
    actor_rollout_ref.nccl_timeout=${nccl_timeout} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    +algorithm.filter_groups.order_by=${order_by} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    +actor_rollout_ref.ref.model.path="${ref_model}" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=32 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=True \
    trainer.test_freq=5 \
    trainer.save_freq=10 \
    trainer.total_epochs=1 \
    trainer.validation_data_dir=${VALIDATION_DATA_DIR} \
    trainer.rollout_data_dir=${ROLLOUT_DATA_DIR} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto
    # trainer.resume_mode='resume_path' \
    # trainer.resume_from_path="${resume_from}"
