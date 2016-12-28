import pycuda.driver as cuda
import pycuda.autoinit
# from pycuda.compiler import SourceModule
from pycuda.driver import module_from_file

import pycuda.gpuarray as gpuarray
import numpy as np

import time

### ------------------------------------------
### start timing the start of the end-to-end processing time
### ------------------------------------------

## load precompiled cubin file
mod = module_from_file("knn_kernels.cubin")

# link to the kernel function

lr_dist = mod.get_function('distKernel')
sort = mod.get_function('sortKernel')


#------------------------------------------------------------------------------
# parameters
#------------------------------------------------------------------------------
neighbors = 3;

# input data

#open file
file_data = open('../data/data1M.txt', 'r')

#grab first line, which has information on the dataset
file_params = file_data.readline().strip("\n").split(" ")

#set dataset parameters variables
numPoints = int(file_params[0])
numDims = int(file_params[1])
numLabels = int(file_params[2])

#make numpy matricies for dataset and labels
data = np.zeros((numPoints, numDims)).astype('f')
labels = np.zeros(numPoints).astype('f')
sortedLabels = np.zeros(numPoints).astype('f')
test_point = np.zeros(numDims).astype('f')


#loop variable to index data and labels
i = 0

#read in the data and labels from the file
for line in file_data:
    split_line = line.strip("\n").split(" ")
    labels[i] = int(split_line[numDims])
    for j in range(0, numDims):
        data[i][j] = float(split_line[j])
    i = i + 1

file_data.close()

start = time.time()

###
## allocate memory on device
###
X_gpu = cuda.mem_alloc(data.nbytes)

labels_gpu = cuda.mem_alloc(labels.nbytes)

sorted_labels_gpu = cuda.mem_alloc(labels.nbytes)

distances_gpu = cuda.mem_alloc(labels.nbytes)

test_point_gpu = cuda.mem_alloc(test_point.nbytes)

###
## transfer data to gpu
###
cuda.memcpy_htod(X_gpu, data)

cuda.memcpy_htod(labels_gpu, labels)

cuda.memcpy_htod(sorted_labels_gpu, labels)

cuda.memcpy_htod(test_point_gpu, test_point)

###
## define kernel configuration
###
blk_size = 16
grd_size = (npoints + blk_size -1) / blk_size

###---------------------------------------------------------------------------
### Run kmeans on gpu
###---------------------------------------------------------------------------
    
    ## run kernel
    distKernel(X_gpu, test_point_gpu, distances_gpu, \
                  np.int32(numPoints), np.int32(1), np.int32(numDims),\
                  block = (blk_size, blk_size, 1), grid = (1, grd_size, 1))

    #sort
    sortKernel(distances_gpu, sorted_labels_gpu,\
               np.int32(numPoints),\
               block = (1, 1, 1), grid = (1, 1, 1))
    
    # copy back
    cuda.memcpy_dtoh(sortedLabels, sorted_labels_gpu)


###----------------------------------------------------------------------------
## end of gpu kmeans
###----------------------------------------------------------------------------

    labelCounts = np.zeros(numLabels).astype('i')

    for i in range(0, neighbors):
        labelCounts[sortedLabels[i]] = labelCounts[sortedLabels[i]] + 1

    output = np.amax(labelCounts)

### ------------------------------------------
### end timing of the end-to-end processing time
### ------------------------------------------
end = time.time()
runtime = end - start

###----------------------------------------------------------------------------
## dump stat
###----------------------------------------------------------------------------


print 'runtime : ' + str(runtime)  + ' s'
print 'label : ' + str(output)
