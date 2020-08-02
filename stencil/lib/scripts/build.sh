#!/bin/bash -l

# set -euo pipefail
source $(dirname ${BASH_SOURCE[0]})/compilers.sh

cd $(dirname ${BASH_SOURCE[0]})/..

root_dir=$(pwd)
build_dir=${1:-./build}
build_type=${2:-RelWithDebInfo}

mkdir -p ${build_dir}
cd ${build_dir}

# TODO load cmake from spack when needed

for compiler in ${compilers[@]}; do
	echo "Building ${build_type} using ${compiler}"
	mkdir -p ${compiler}
	cd ${compiler}

	load_compiler ${compiler}
	# export CRAYPE_LINK_TYPE=static

	cmake -D CMAKE_BUILD_TYPE=${build_type} ${root_dir}
	cmake --build .

	cd ..
done

# Rust
mkdir -p rustc
cd rustc

load_compiler cray
# export CRAYPE_LINK_TYPE=dynamic
RUSTFLAGS="-C target-cpu=${CRAY_CPU_TARGET} -C relocation-model=dynamic-no-pic" \
	cargo build \
		-Z unstable-options \
		--manifest-path ${root_dir}/Cargo.toml \
		--profile ${build_type,,} \
		--target-dir . \
		--out-dir rs

cd ..
