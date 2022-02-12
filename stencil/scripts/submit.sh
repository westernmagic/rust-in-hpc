#!/bin/bash -l
#SBATCH --job-name        stencil
#SBATCH --output          %x.out
#SBATCH --error           %x.err
#SBATCH --open-mode       truncate
#SBATCH --nodes           4
#SBATCH --constraint      gpu
#SBATCH --time            00:20:00
#SBATCH --partition       debug
#SBATCH --ntasks-per-core 1
#SBATCH --hint            nomultithread
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task   12

module load cce
module load gcc
module load intel
module load pgi
module load cudatoolkit
module load GREASY

root_path=${SLURM_SUBMIT_DIR}
target_path=${root_path}/target/release

export GREASY_NWORKERS_PER_NODE=${SLURM_NTASKS_PER_NODE}
export RAYON_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_TARGET_OFFLOAD="MANDATORY"
export CRAY_CUDA_MPS=1

export LD_LIBRARY_PATH=$(echo ${target_path}/build/stencil-*/out/{gnu,intel,cray,pgi,rustc}/{cpp,f,rs} | sed -e 's/ /:/g'):${LD_LIBRARY_PATH}

mkdir -p data

export RAYON_NUM_THREADS=1
export OMP_NUM_THREADS=1
export GREASY_LOGFILE=${SLURM_JOB_NAME}_seq_greasy.log
greasy scripts/seq.txt

export RAYON_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export GREASY_LOGFILE=${SLURM_JOB_NAME}_cpu_greasy.log
greasy scripts/cpu.txt

export RAYON_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export GREASY_LOGFILE=${SLURM_JOB_NAME}_gpu_greasy.log
greasy scripts/gpu.txt
