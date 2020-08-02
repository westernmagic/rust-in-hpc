#!/bin/bash

module load gcc
module load intel
module load cce
module load pgi

root_path=$(dirname ${BASH_SOURCE[0]})/..
target_path=${root_path}/../target/release

export LD_LIBRARY_PATH=$(echo ${target_path}/build/stencil-*/out/{gnu,intel,cray,pgi,rustc}/{cpp,f,rs} | sed -e 's/ /:/g'):${LD_LIBRARY_PATH}


${target_path}/stencil $*
