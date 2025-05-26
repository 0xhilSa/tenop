#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdio.h>

int main(){
  int device_count = 0;
  cudaError_t error = cudaGetDeviceCount(&device_count);
  if(error != cudaSuccess){
    printf("CUDA Error: %s\n", cudaGetErrorString(error));
    return 1;
  }
  printf("Found %d CUDA device(s) \n", device_count);
  for(int i = 0; i < device_count; ++i){
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device %d: %s\n", i, prop.name);
  }
  int chosed = 0;
  if(chosed >= device_count){
    printf("Invalid device index\n");
    return 1;
  }
  cudaSetDevice(chosed);
  printf("Using CUDA device %d\n", chosed);
  return 0;
}

