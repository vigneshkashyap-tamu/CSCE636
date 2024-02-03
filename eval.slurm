#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --time=1:00:00
#SBATCH --mem=40G
#SBATCH --output=./job_output/%x_%j.log
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=a100:1
#SBATCH --partition=gpu

#First Executable Line
source ~/.bashrc
conda activate csce689_proj

export PYTHONPATH="$PYTHONPATH:./bimodal_exps"
export HUGGINGFACE_HUB_CACHE='./checkpoints/huggingface'

data_path=./datasets/
imagenet_val_path=./datasets/imagenet/val

data=cc3m

if [ "${data}" = 'cc3m' ]; then
    train_image_root="['cc3m']"
    train_file="['cc3m/cc3m_train/{00000..00331}.tar']"
    train_num_samples=2723200
    train_num_samples_total=3400000
elif [ "${data}" = 'cc12m' ]; then
    train_image_root="['cc12m']"
    train_file="['cc12m/cc12m/{00000..01242}.tar']"
    train_num_samples=9184256
    train_num_samples_total=12423374
elif [ "${data}" = 'coco' ]; then
    train_image_root="['coco/train2014']"
    train_file="['coco/annotations/captions_train2014.json']"
    train_num_samples=420000
    train_num_samples_total=420000
fi

frac=1.0
gamma=0.8
rho=8.0
ita_type=isogclr_new
# ita_type=clip
lr='2e-4'
weight_decay=0.02
eta_init='3e-2'
epochs=30

CUDA_VISIBLE_DEVICES='0' python ./bimodal_exps/clip.py \
    --data_path ${data_path} \
    --train_file ${train_file} \
    --train_image_root ${train_image_root} \
    --output_dir output/"${SLURM_JOB_NAME}" \
    --init_model \
    --use_amp \
    --ita_type "${ita_type}" \
    --tau_init 0.01 \
    --sogclr_gamma ${gamma} \
    --eta_init "${eta_init}" --sched cosine \
    --epochs 1 --lr "${lr}" \
    --embed_dim 256 \
    --batch_size_train 256 \
    --rho_I "${rho}" --rho_T "${rho}" \
    --train_frac ${frac} \
    --zs_dataset imagenet \
    --zs_datafolder ${imagenet_val_path} \
    --train_num_samples "${train_num_samples}" \
    --train_num_samples_total "${train_num_samples_total}" \
    --weight_decay "${weight_decay}" \
    --evaluate --checkpoint "output/"${SLURM_JOB_NAME:5}"/checkpoint_1.pth"
