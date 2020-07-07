#include <iostream>
#include <math.h>
#include "slicing.h"



// Kernel function to add the elements of two arrays
__global__
void add(dim3 blockOffset, dim3 realGridDim, int n, float *x, float *y)
{
  int rBlockIdx = blockOffset.x + blockIdx.x;
  int rBlockIdy = blockOffset.y + blockIdx.y;
  int rBlockIdz = blockOffset.z + blockIdx.z;

  int index = rBlockIdx * blockDim.x + threadIdx.x;
  int stride = blockDim.x * realGridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}

int main(void)
{
  int N = 1<<27;
  float *x, *y;
  

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Run kernel on 1M elements on the GPU
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  dim3 nBlocks = {numBlocks, 1, 1};

  SLICER(nBlocks, blockSize, add, N, x, y);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  cudaFree(x);
  cudaFree(y);
  
  return 0;
}
