import pycuda.driver as cuda
import pycuda.autoinit  # noqa
#from pycuda.compiler import SourceModule
from pycuda.compiler import compile

test_kernel = compile(
"""
   #include <thrust/extrema.h>
   #include <thrust/device_ptr.h>
   #include <thrust/execution_policy.h>
   #include <thrust/sort.h>
const int BLOCKSIZE = 32;

extern "C"{   
__global__ void distKernel(float *devA, float *devB, float *devC, int rows, int cols, int K)
{
int idy = threadIdx.y + blockIdx.y * blockDim.y;
int idx = threadIdx.x + blockIdx.x * blockDim.x;

__shared__ float tileA[BLOCKSIZE][BLOCKSIZE];
__shared__ float tileB[BLOCKSIZE];

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
if ((threadIdx.y + BLOCKSIZE*iter < K)){
tileB[threadIdx.y] = devB[(threadIdx.y + BLOCKSIZE*iter)*cols];
}
else {
tileB[threadIdx.y] = 0.0;
}

//Sync to ensure that all of the data has been grabbed for the tiles in this warp
__syncthreads();

//Sum the squared distance between the terms
for (int i = 0; i < BLOCKSIZE; i++){
sum += (tileA[threadIdx.y][i] - tileB[i])*(tileA[threadIdx.y][i] - tileB[i]);
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

__global__ void getKLabels(float* dist, int* labels, int* kToReturn, int npoints, int k){  
    thrust::device_ptr<float> A(dist); 
    thrust::device_ptr<float> offset; float bigNum = 10000000000000.0;

    for (int i = 0; i < k; i++){
        offset = thrust::min_element(thrust::device, A, A + npoints);
        *(dist+(offset-A)) = bigNum;
	kToReturn[i] = *(labels+(offset-A));
    }
}
__global__ void sort(float* dist, int* labels, int* kLabels,  int npoints, int k){  
    thrust::device_ptr<float> A(dist); thrust::device_ptr<int> B(labels);
    //thrust::sort_by_key(thrust::device, A, A + npoints, B);
    thrust::sort(thrust::seq, dist, dist + npoints);

    for (int i = 0; i < k; i++){
        kLabels[i] = labels[i];
    }
}

 }
    """, no_extern_c=True)

with open("knn_kernels.cubin", "wb") as file:
    file.write(test_kernel)
