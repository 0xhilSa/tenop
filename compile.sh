#!/bin/bash

CC=gcc
NVCC=nvcc
PYTHON_VERSION=3.10
PYTHON_INCLUDE=$(python${PYTHON_VERSION}-config --includes)
PYTHON_LIBS=$(python${PYTHON_VERSION}-config --ldflags | sed 's@/usr/lib[^ ]*libdl.a@@g')
CUDA_PATH=/usr/local/cuda

HOST_SRC="./tenop/engine/cpu.c"
HOST_OUT="./tenop/engine/cpu.so"
CUDA_SRC="./tenop/engine/cuda.cu"
CUDA_OUT="./tenop/engine/cuda.so"

CPU_OPS_SRC="./tenop/engine/cpu_ops.c"
CPU_OPS_OUT="./tenop/engine/cpu_ops.so"
CUDA_OPS_SRC="./tenop/engine/cuda_ops.cu"
CUDA_OPS_OUT="./tenop/engine/cuda_ops.so"

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

$CC -Wall -shared -fPIC $CPU_OPS_SRC -o $CPU_OPS_OUT $PYTHON_INCLUDE $PYTHON_LIBS &

compile_pid=$!
spinner $compile_pid
wait $compile_pid

$NVCC -shared -Xcompiler -fPIC -I$CUDA_PATH/include $PYTHON_INCLUDE \
    -o $CUDA_OUT -L$CUDA_PATH/lib64 -lcudart $CUDA_SRC &

compile_pid=$!
spinner $compile_pid
wait $compile_pid

$NVCC -shared -Xcompiler -fPIC -I$CUDA_PATH/include $PYTHON_INCLUDE \
    -o $CUDA_OUT -L$CUDA_PATH/lib64 -lcudart $CUDA_SRC &

compile_pid=$!
spinner $compile_pid
wait $compile_pid

$NVCC -shared -Xcompiler -fPIC -I$CUDA_PATH/include $PYTHON_INCLUDE \
    -o $CUDA_OPS_OUT -L$CUDA_PATH/lib64 -lcudart $CUDA_OPS_SRC &

compile_pid=$!
spinner $compile_pid
wait $compile_pid

echo -ne "\r\033[KCompiled Successfully!\n"
