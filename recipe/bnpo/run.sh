export VLLM_ATTENTION_BACKEND=XFORMERS

# data
train_files="/data/DigitalLearningGmbH/MATH-lighteval/train_repeated.parquet"
val_files="['/data/HuggingFaceH4/MATH-500/test_repeated.parquet',  '/data/math-ai/amc23/test_repeated.parquet', '/data/math-ai/aime24/test_repeated.parquet', '/data/math-ai/aime25/test_repeated.parquet']"
prompt_key="prompt"
reward_fn_key="data_source"
max_prompt_length=$((1024 * 1))
max_response_length=$((1024 * 3))
train_batch_size=32
filter_overlong_prompts=False
truncation="left"

# actor_rollout_ref
# model
model_path="Qwen/Qwen2.5-Math-7B"
enable_gradient_checkpointing=True
use_remove_padding=True
# actor
ppo_mini_batch_size=32
use_dynamic_bsz=True
max_token_len_per_gpu=32768 # $(((max_prompt_length + max_response_length)))
grad_clip=1.0
clip_ratio_low=0.2
clip_ratio_high=0.28
clip_ratio_c=10.0
loss_agg_mode="seq-mean-token-sum"
entropy_coeff=0.0
use_kl_loss=False
kl_loss_coef=0.0
ppo_epochs=1
ulysses_sequence_parallel_size=1
lr=1e-6
lr_warmup_steps=10
weight_decay=0.0
offload=False
# rollout
temperature=1.0
top_p=1.0
gpu_memory_utilization=0.8
tensor_model_parallel_size=1
rollout_n=16
val_kwargs_temperature=0.6
val_kwargs_top_p=1.0
val_kwargs_n=1
do_sample=True

# reward_model
reward_manager="bnpo"

# algorithm
adv_estimator="bnpo"
use_kl_in_reward=False
kl_coef=0.0
use_format_reward=False
use_advantage_decomposition=False

# trainer
total_epochs=5
project_name='BNPO'
experiment_name=-usead${use_advantage_decomposition}-format${use_format_reward}-adv${adv_estimator}
log_val_generations=10
nnodes=1
n_gpus_per_node=4
save_freq=-1
test_freq=16
default_local_dir="checkpoints/${project_name}/${experiment_name}"


nohup ray job submit --no-wait \
    --runtime-env-json='{
        "working_dir": "'${PWD}'"
    }' \
    -- python -m recipe.bnpo.src.main_ppo \
    data.train_files="${train_files}" \
    data.val_files="${val_files}" \
    data.prompt_key="${prompt_key}" \
    data.reward_fn_key="${reward_fn_key}" \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_batch_size} \
    data.filter_overlong_prompts=${filter_overlong_prompts} \
    data.truncation=${truncation} \
    actor_rollout_ref.model.path="${model_path}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=${enable_gradient_checkpointing} \
    actor_rollout_ref.model.use_remove_padding=${use_remove_padding} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${max_token_len_per_gpu} \
    actor_rollout_ref.actor.grad_clip=${grad_clip} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=${clip_ratio_c} \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.ppo_epochs=${ppo_epochs} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${ulysses_sequence_parallel_size} \
    actor_rollout_ref.actor.optim.lr=${lr} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=${lr_warmup_steps} \
    actor_rollout_ref.actor.optim.weight_decay=${weight_decay} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${tensor_model_parallel_size} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_token_len_per_gpu} \
    actor_rollout_ref.rollout.n=${rollout_n} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_kwargs_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_kwargs_top_p} \
    actor_rollout_ref.rollout.val_kwargs.n=${val_kwargs_n} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=${do_sample} \
    reward_model.reward_manager=${reward_manager} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.use_advantage_decomposition=${use_advantage_decomposition} \
    algorithm.use_format_reward=${use_format_reward} \
    trainer.total_epochs=${total_epochs} \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${experiment_name}" \
    trainer.log_val_generations="${log_val_generations}" \
    trainer.nnodes="${nnodes}" \
    trainer.n_gpus_per_node="${n_gpus_per_node}" \
    trainer.save_freq="${save_freq}" \
    trainer.test_freq="${test_freq}" \
    trainer.default_local_dir="${default_local_dir}" > log.txt 2>&1 &
