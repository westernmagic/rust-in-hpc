#!/bin/bash -l
#SBATCH --job-name        benchmark
#SBATCH --output          %x.out
#SBATCH --error           %x.err
#SBATCH --open-mode       truncate
#SBATCH --nodes           16
#SBATCH --constraint      gpu
#SBATCH --time            08:00:00
#SBATCH --partition       normal
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
export OMP_TARGET_OFFLOAD="MANDATORY"
export CRAY_CUDA_MPS=1

export LD_LIBRARY_PATH=$(echo ${target_path}/build/stencil-*/out/{gnu,intel,cray,pgi,rustc}/{cpp,f,rs} | sed -e 's/ /:/g'):${LD_LIBRARY_PATH}

mkdir -p ${SCRATCH}/data
ln -s ${SCRATCH}/data ./data

for N in 128 256 512 1024; do
	export NX=${N}
	export NY=${N}

	export RAYON_NUM_THREADS=1
	export OMP_NUM_THREADS=1
	export GREASY_LOGFILE=${SLURM_JOB_NAME}_seq_${N}_greasy.log
	greasy scripts/seq.txt

	for p in 1 2 4 8 12; do
		export RAYON_NUM_THREADS=${p}
		export OMP_NUM_THREADS=${p}
		export GREASY_LOGFILE=${SLURM_JOB_NAME}_cpu_${N}_${p}_greasy.log
		greasy scripts/cpu.txt
	done

	export RAYON_NUM_THREADS=12
	export OMP_NUM_THREADS=12
	export GREASY_LOGFILE=${SLURM_JOB_NAME}_gpu_${N}_greasy.log
	greasy scripts/gpu.txt
done
