#import gpuadder
import numpy as np
from knn import knnGPU
import time

#open file
file_data = open('data/data100k10.txt', 'r')

#grab first line, which has information on the dataset
file_params = file_data.readline().strip("\n").split(" ")

#set dataset parameters variables
numPoints = int(file_params[0])
numDims = int(file_params[1])
numLabels = int(file_params[2])

#make numpy matricies for dataset and labels
data = np.zeros((numPoints*numDims),  dtype=np.float32)
#data = np.zeros((numPoints, numDims),  dtype=np.float32)
labels = np.zeros(numPoints,  dtype=np.int32)

#loop variable to index data and labels
i = 0

#read in the data and labels from the file
for line in file_data:
    split_line = line.strip("\n").split(" ")
    labels[i] = int(split_line[numDims])
    for j in range(0, numDims):
        data[i*numDims + j] = float(split_line[j])
        #data[i][j] = float(split_line[j])
    i = i + 1

num_neighbors = 3
point = np.zeros((numDims), dtype=np.float32);
#point = np.zeros((1, numDims), dtype=np.float32);
point[0] = 10.0
# start timer
start = time.time()

mydata = knnGPU(data, labels, num_neighbors, numPoints, numDims, numLabels)

classification = mydata.classify(point)

# end the timer
end = time.time()

runtime = end - start

print str(end - start) +  ' s'
print str(classification)


