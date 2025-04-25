#!/bin/bash

CC=gcc
NVCC=nvcc
PYTHON_VERSION=3.10
PYTHON_INCLUDE=$(python${PYTHON_VERSION}-config --includes)
PYTHON_LIBS=$(python${PYTHON_VERSION}-config --ldflags | sed 's@/usr/lib[^ ]*libdl.a@@g')
CUDA_PATH=/usr/local/cuda

#HOST_SRC="./rush/csrc/cpu_core.c"
#CUDA_SRC="./rush/csrc/cuda_core.cu"
#HOST_OUT="./rush/csrc/cpu_core.so"
#CUDA_OUT="./rush/csrc/cuda_core.so"

HOST_SRC="./rush/engine/cpu_core.c"
HOST_OUT="./rush/engine/cpu_core.so"
CUDA_SRC="./rush/engine/cuda_core.cu"
CUDA_OUT="./rush/engine/cuda_core.so"

spinner(){
  local pid=$1
  local delay=0.1
  local spin='|/-\'
  local i=0

  while kill -0 $pid 2>/dev/null; do
    printf "\rCompiling C and CUDA source file %s" "${spin:i++%4:1}"
    sleep $delay
  done
}

$CC -Wall -shared -fPIC $HOST_SRC -o $HOST_OUT $PYTHON_INCLUDE $PYTHON_LIBS &

compile_pid=$!
spinner $compile_pid
wait $compile_pid

$NVCC -shared -Xcompiler -fPIC -I$CUDA_PATH/include $PYTHON_INCLUDE \
    -o $CUDA_OUT -L$CUDA_PATH/lib64 -lcudart $CUDA_SRC &

compile_pid=$!
spinner $compile_pid
wait $compile_pid

echo -ne "\r\033[KCompiled Successfully!\n"
