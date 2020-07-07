#include <iostream>
#include <math.h>
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

int main(void){
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
    
    // without slicing this call would have been MatAdd<<<numBlocks, threadsPerBlock>>>(N, a, b, c) 
    SLICER(numBlocks, threadsPerBlock, MatAdd, N, a, b, c);

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
  
    return 0;
}
