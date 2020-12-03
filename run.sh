#!/bin/bash
#SBATCH -A gpu
#SBATCH -p highGPU
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH -o /scratch/visual/ashestak/cluster_jobs/detr_slice2d_output%j.out
#SBATCH --job-name=detr2d

rootdir=/scratch/visual/ashestak
datadir=$rootdir/oai/v00/numpy/full/
num_workers=5
num_classes=1
num_queries=1
max_epochs=500
batch_size=16
weights_save_path=checkpoints/
experiment_name=2DSlice_Detr
overfit_batches=10
gpus=1
limit_train_batches=500
limit_val_batches=50

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
	--gpus $gpus \
	--no_aux_loss \
