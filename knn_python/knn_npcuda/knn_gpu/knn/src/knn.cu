#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/sort.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <stdio.h>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <cmath>

#include <kernel.cu>
#include <knn.hh>
using namespace std;

void sortArr(float* key, int* value, int numPoints){
	thrust::device_ptr<float> A(key); thrust::device_ptr<int> B(value);
	thrust::sort_by_key(A, A + numPoints, B);
}

knnGPU::knnGPU(float* data_in, int* labels_in, int neighbors, int points, int features, int label){

    //Store params for this knn set up
    k = neighbors;
    npoints = points;
    nfeatures = features;
    nlabels = label;

    data_bytes = npoints*nfeatures*sizeof(float);
    distance_bytes = npoints*sizeof(float);
    label_bytes = npoints*sizeof(int);

    if(data != NULL) cudaFree(data);
    cudaMallocManaged((void **)&data, data_bytes);
    cudaMemcpy(data, data_in, data_bytes, cudaMemcpyHostToDevice);

    if(labels != NULL) cudaFree(labels);
    cudaMallocManaged((void **)&labels, label_bytes);
    cudaMemcpy(labels, labels_in, label_bytes, cudaMemcpyHostToDevice);

    if(sortedLabels != NULL) cudaFree(sortedLabels);
    cudaMallocManaged((void **)&sortedLabels, label_bytes);
    cudaMemcpy(sortedLabels, labels_in, label_bytes, cudaMemcpyHostToDevice);

    if(distances != NULL) cudaFree(data);
    cudaMallocManaged((void **)&distances, distance_bytes);

    blocksize = 16;
    blkDim = dim3(blocksize, blocksize, 1);
    grdDim = dim3(1, BLK(npoints, blocksize), 1);

}

knnGPU::~knnGPU(){
    Cleanup();
}

int knnGPU::classify(float* point_in){

    float *point;
    size_t point_bytes = nfeatures*sizeof(float);

    cudaMallocManaged((void **)&point, point_bytes);
    cudaMemcpy(point, point_in, point_bytes, cudaMemcpyHostToDevice);

    distKernel<<<grdDim, blkDim>>>(data, point, distances, npoints, 1, nfeatures);

   // thrust::device_ptr<float> dist = thrust::device_pointer_cast(distances);
   // thrust::device_ptr<int> sl(sortedLabels);
   //thrust::sort_by_key(dist, dist + npoints, sl);
   
    sortArr(distances, sortedLabels, npoints);

    int *labelCounts = new int[nlabels];
    int *kLabels = new int[k];
    cudaMemcpy(kLabels, sortedLabels, k*sizeof(int), cudaMemcpyDeviceToHost);	
    
   // for (int i = 0; i < k; i++){
//	kLabels[i] = sl[i];
//    }

    for (int i = 0; i < nlabels; i++){
        labelCounts[i] = 0;
    }

    for (int i = 0; i < k; i++){
        labelCounts[kLabels[i]] += 1;
    }

    int* outputLabel = thrust::max_element(thrust::host, labelCounts, labelCounts + k);
    int output = outputLabel - labelCounts;

    cudaFree(point);

    cudaMemcpy(sortedLabels, labels, label_bytes, cudaMemcpyDeviceToDevice);
    return output;
}

void knnGPU::Cleanup(){
    if(data != NULL) cudaFree(data);
    if(labels != NULL) cudaFree(labels);
    if(distances != NULL) cudaFree(distances);
}
