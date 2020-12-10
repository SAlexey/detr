#!/bin/bash
#SBATCH -A gpu
#SBATCH -p highGPU
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH -o /scratch/visual/ashestak/cluster_jobs/detr_slice2d_output%j.out
#SBATCH --job-name=pretrain

rootdir=/scratch/visual/ashestak
datadir=$rootdir/oai/v00/numpy/full/train
num_workers=4
num_classes=3
num_queries=1
max_epochs=500
batch_size=1
weights_save_path=checkpoints/
experiment_name=BackbonePretrain_500
overfit_batches=10
gpus=1
limit_train_batches=100
limit_val_batches=10

export https_proxy="http://130.73.108.40:8080"
export http_proxy="http://130.73.108.40:8080"

exec python pretrain.py --datadir $datadir \
    --batch_size $batch_size \
    --experiment_name $experiment_name \
    --num_workers $num_workers \
    --gpus $gpus \
	
