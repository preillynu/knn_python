import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNN
import time

#open file
file_data = open('../data/data100k.txt', 'r')

#grab first line, which has information on the dataset
file_params = file_data.readline().strip("\n").split(" ")

#set dataset parameters variables
numPoints = int(file_params[0])
numDims = int(file_params[1])
numLabels = int(file_params[2])

#make numpy matricies for dataset and labels
data = np.zeros((numPoints, numDims), dtype = np.float32)
labels = np.zeros(numPoints, dtype = np.float32)
test_point = np.zeros(numDims, dtype = np.float32)
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

start_time = time.time()

#run knn using the sklearn library
neighbors = KNN(n_neighbors = 3, n_jobs=1)
neighbors.fit(data, labels)
neighbors.predict([test_point])
end_time = time.time() - start_time

print "Time take for knn: ", end_time, " seconds"

outfile = open('out1/out100k.txt', 'a')
outfile.write(str(end_time))
outfile.write('\n')
outfile.close()

#print sample results for some new points with label 0 and label 1
#print(neighbors.predict([[0.1, 10.2]]))
#print(neighbors.predict([[10.1, 0.2]]))
