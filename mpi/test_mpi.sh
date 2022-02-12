#!/bin/bash -l
#SBATCH --job-name="test_mpi"
#SBATCH --time=00:15:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=36
#SBATCH --cpus-per-task=1
#SBATCH --partition=debug
#SBATCH --constraint=mc
#SBATCH --hint=nomultithread
set -euo pipefail
IFS=$'\n\t'

cargo build --release --examples
for example in $(ls -f examples/*.rs); do
	example=${example##*/}
	example=${example%%.rs}
	case "${example}" in
		"comm_name" )
			echo "Skipping ${example}..."
			;;
		* )
			echo "Running ${example}..."
			srun ./target/release/examples/${example}
			;;
	esac
done
