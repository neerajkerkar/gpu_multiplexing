#include <iostream>
#include <math.h>
#include <pthread.h>
#include "slicing.h"


// Kernel definition needs to have 2 extra fields blockOffset and realGridDim
// within kernel use blockOffset + blockId instead of blockId directly
// use realGridDim instead of gridDim
__global__ void MatAdd(dim3 blockOffset, dim3 realGridDim, int N, float* A, float* B, float* C)
{
    int rBlockIdx = blockOffset.x + blockIdx.x;
    int rBlockIdy = blockOffset.y + blockIdx.y;
    int rBlockIdz = blockOffset.z + blockIdx.z;

    int i = rBlockIdx * blockDim.x + threadIdx.x;
    int j = rBlockIdy * blockDim.y + threadIdx.y;
    int k = i+j*N;
    if (i < N && j < N)
        C[k] = A[k] + B[k];
}

// Kernel function to add the elements of two arrays
__global__
void VecAdd(dim3 blockOffset, dim3 realGridDim, int n, float *x, float *y)
{
  int rBlockIdx = blockOffset.x + blockIdx.x;
  int rBlockIdy = blockOffset.y + blockIdx.y;
  int rBlockIdz = blockOffset.z + blockIdx.z;

  int index = rBlockIdx * blockDim.x + threadIdx.x;
  int stride = blockDim.x * realGridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}

__global__ void dummy_kernel(int *a){
  a+=1;
}

void* launch_matadd(void* dummy){
    int N = 1<<13;
    float *a,*b,*c;
  
    cudaMallocManaged(&a, N*N*sizeof(float));
    cudaMallocManaged(&b, N*N*sizeof(float));
    cudaMallocManaged(&c, N*N*sizeof(float));

    for (int i = 0; i < N*N; i++) {
      a[i] = 1.0f;
      b[i] = 2.0f;
    }

    dim3 threadsPerBlock ={16, 16, 1};
    dim3 numBlocks={N / threadsPerBlock.x, N / threadsPerBlock.y, 1};
    
    //cudaDeviceSynchronize();
    
    dim3 blockOff = {0,0,0};
    MatAdd<<<numBlocks, threadsPerBlock>>>(blockOff, numBlocks, N, a, b, c);

    cudaDeviceSynchronize();

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N*N; i++)
      maxError = fmax(maxError, fabs(c[i]-3.0f));
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
  
    return NULL;
}

void* launch_vecadd(void* dummy)
{
  int N = 1<<26;
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

  dim3 blockOff = {0,0,0};
  VecAdd<<<nBlocks, blockSize>>>(blockOff, nBlocks, N, x, y);

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
  
  return NULL;
}

int main(void){
  const int num_threads = 2;
  pthread_t threads[num_threads];
  /*int a=1;
  dummy_kernel<<<1,1>>>(&a);
  dim3 blockOff = {0,0,0};
  dim3 grid = {0,0,0};
  MatAdd<<<1,1>>>(blockOff,grid,0,NULL,NULL,NULL);*/
  for (int i = 0; i < num_threads; i++) {
    void * (*launch_kernel)(void *);
    if(i%2==1) launch_kernel = launch_matadd;
    else launch_kernel = launch_vecadd;
    if (pthread_create(&threads[i], NULL, launch_kernel, 0)) {
      fprintf(stderr, "Error creating threadn");
      return 1;
    }
  }
  for (int i = 0; i < num_threads; i++) {
    if(pthread_join(threads[i], NULL)) {
      fprintf(stderr, "Error joining threadn");
      return 2;
    }
  }
  return 0;
}


