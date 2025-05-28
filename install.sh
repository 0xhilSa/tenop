#!/bin/bash

CC=gcc
NVCC=nvcc
PYTHON_VERSION=3.10
PYTHON_INCLUDE=$(python${PYTHON_VERSION}-config --includes)
PYTHON_LIBS=$(python${PYTHON_VERSION}-config --ldflags | sed 's@/usr/lib[^ ]*libdl.a@@g')
CUDA_PATH=/usr/local/cuda

HOST_SRC="./cuten/engine/cpu.c"
HOST_OUT="./cuten/engine/cpu.so"
CUDA_SRC="./cuten/engine/cuda.cu"
CUDA_OUT="./cuten/engine/cuda.so"

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

AUTO_YES=false

if python3 -c "import cuten" 2>/dev/null; then
  if [ "$AUTO_YES" = true ]; then
    echo "Reinstalling cuten ..."
    pip uninstall -y cuten
  else
    read -p "cuten is already installed. Do you want to reinstall it? (y/N)" choice
    case "$choice" in
      y|Y)
        echo "Reinstalling cuten ..."
        pip uninstall -y cuten
        ;;
      *)
        echo "Aborting installation"
        exit 1
        ;;
    esac
  fi
fi

python3 -m build
cd dist/ && pip install *.whl && cd ..
rm -rf dist/ build/ cuten.egg-info/
