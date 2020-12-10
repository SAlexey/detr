#!/bin/bash
#SBATCH -A gpu
#SBATCH -p highGPU
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH -o /scratch/visual/ashestak/cluster_jobs/detr_slice2d_output%j.out
#SBATCH --job-name=detr2d

rootdir=/scratch/visual/ashestak
datadir=$rootdir/oai/v00/numpy/full/
num_workers=5
num_classes=91
num_queries=100
max_epochs=500
batch_size=16
weights_save_path=checkpoints/
experiment_name=2D_aug
backbone_checkpoint_path=/scratch/visual/ashestak/detr/checkpoints/BackbonePretrain_500/epoch=24_validation_loss=0.030.ckpt
overfit_batches=10
gpus=1
limit_train_batches=100
limit_val_batches=10
lr_backbone=0.0001

export https_proxy="http://130.73.108.40:8080"
export http_proxy="http://130.73.108.40:8080"

exec python main.py --datadir $datadir \
	--limit_train_batches $limit_train_batches \
	--limit_val_batches $limit_val_batches \
	--num_workers $num_workers \
	--num_classes $num_classes \
	--num_queries $num_queries \
	--max_epochs $max_epochs \
	--experiment_name $experiment_name \
	--batch_size $batch_size \
	--gpus $gpus \
	--no_aux_loss \
