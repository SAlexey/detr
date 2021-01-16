#!/bin/bash
#SBATCH -A gpu
#SBATCH -p highGPU
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH -o /scratch/visual/ashestak/cluster_jobs/d2d_100_out_%j.out
#SBATCH --job-name=d2d-100


export https_proxy="http://130.73.108.40:8080"
export http_proxy="http://130.73.108.40:8080"

exec python main.py
