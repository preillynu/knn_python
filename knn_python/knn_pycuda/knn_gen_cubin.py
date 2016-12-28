import pycuda.driver as cuda
import pycuda.autoinit  # noqa
from pycuda.compiler import SourceModule
#from pycuda.compiler import compile

test_kernel = SourceModule(
"""
    #include <thrust/sort.h>
    #include <thrust/device_ptr.h>

extern "C" {    
    const int BLOCKSIZE = 32;
   
     __global__ void distKernel(float *devA, float *devB, float *devC, int rows, int cols, int K)
    {
    //Get the thread's x and y locations for its run
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    
    //Allocate` shared memory to hold parts of A and B
    __shared__ float tileA[BLOCKSIZE][BLOCKSIZE];
    __shared__ float tileB[BLOCKSIZE][BLOCKSIZE];
    
    //Use sum to get the result for a specific element
    float sum = 0.0;
    
    //Use iter to see if the loop should be run again
    int iter = 0;
    
    do{
    //Check if the x thread falls within bounds of the matrices
    if ((idy < rows) && (threadIdx.x + BLOCKSIZE*iter < K)){
    tileA[threadIdx.y][threadIdx.x] = devA[threadIdx.x + idy*K + BLOCKSIZE*iter];
    }
    else {
    tileA[threadIdx.y][threadIdx.x] = 0.0;
    }
    
    //Check if the y thread falls within bounds of the matrices
    if ((threadIdx.y + BLOCKSIZE*iter < K) && (idx < cols)){
    tileB[threadIdx.y][threadIdx.x] = devB[idx + (threadIdx.y + BLOCKSIZE*iter)*cols];
    }
    else {
    tileB[threadIdx.y][threadIdx.x] = 0.0;
    }
    
    //Sync to ensure that all of the data has been grabbed for the tiles in this warp
    __syncthreads();
    
    //Sum the squared distance between the terms
    for (int i = 0; i < BLOCKSIZE; i++){
    sum += (tileA[threadIdx.y][i] - tileB[i][threadIdx.x])*(tileA[threadIdx.y][i] - tileB[i][threadIdx.x]);
    }
    
    //Iterate the number done
    iter++;
    
    //Sync the threads again to ensure they have all done their work before going through the loop to get data
    __syncthreads();
    
    //Check if the tiles have covered all of C
    } while (BLOCKSIZE*iter < K);
    
    	//If the thread falls within the matrix C, fill in its element, scaled by alpha and beta
    	if ((idy < rows) && (idx < cols)){
    		devC[idx + idy*cols] = sum;
    	}
    }
    
    __global__ void sortKernel(float *devA, float *devB, int npoints)
    {
    	int idx = threadIdx.x + blockIdx.x * blockDim.x;
    	if (idx == 0){
    		thrust::device_ptr<float> A(devA); thrust::device_ptr<float> B(devB);
    		thrust::sort_by_key(A, A + npoints, B);
	}
    }
}
    """, no_extern_c = True, include_dirs=[])

with open("knn_kernels.cubin", "wb") as file:
    file.write(test_kernel)
