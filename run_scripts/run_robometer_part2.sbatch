#!/bin/bash
#SBATCH -J rbm_robometer_4gpu
#SBATCH --gres=gpu:4               # number of GPUs to reserve
#SBATCH --cpus-per-task=32        # CPUs for data loading (8 per GPU is reasonable)
#SBATCH --mem=800G                 # Memory allocation (adjust based on your cluster)
#SBATCH --time=24:00:00            # Time limit (adjust as needed)
#SBATCH --output=slurm_logs/slurm_%j.out # Standard output
#SBATCH --error=slurm_logs/slurm_%j.err  # Standard error
#SBATCH --exclude=g002

source ~/.bashrc


# Exit on error
set -e

ulimit -n 65536

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

# Set environment variables
export BS=16

# Run training with accelerate
# Using FSDP config with 4 processes for 4 GPUs
# Override config for uniform_sample progress-only training with 2 frames

uv run accelerate launch --config_file robometer/configs/distributed/ddp.yaml \
  --num_processes=4 \
  --fsdp_transformer_layer_cls_to_wrap="Qwen3VLVisionBlock,Qwen3VLTextDecoderLayer" \
  train.py \
  data.train_datasets=[rbm-1m-id] \
  data.eval_datasets=[rbm-1m-ood] \
  data.max_frames=8 \
  model.train_progress_head=true \
  model.train_preference_head=true \
  training.exp_name="rbm_retrain_4b_16bs_4gpu_15ksteps_part2" \
  training.per_device_train_batch_size=${BS} \
  training.per_device_eval_batch_size=${BS} \
  training.overwrite_output_dir=true \
  training.max_steps=15000 \
  custom_eval.reward_alignment=[rbm-1m-ood] \
  custom_eval.policy_ranking=[rbm-1m-ood] \
  custom_eval.confusion_matrix=[rbm-1m-ood] \
  logging.save_best.metric_names=[eval_p_rank/kendall_last_utd_so101_clean_top,eval_p_rank/kendall_last_usc_xarm,eval_p_rank/kendall_last_usc_franka,eval_p_rank/kendall_last_rfm_new_mit_franka_nowrist,eval_p_rank/kendall_last_usc_trossen] \
  logging.save_best.greater_is_better=[True,True,True,True,True]

# Print completion time
echo "End Time: $(date)"
echo "Job completed successfully"
