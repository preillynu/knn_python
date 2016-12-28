import numpy as np
cimport numpy as np

assert sizeof(int)   == sizeof(np.int32_t)
assert sizeof(float) == sizeof(np.float32_t)

cdef extern from "src/knn.hh":
    cdef cppclass C_knnGPU "knnGPU":
        C_knnGPU (np.float32_t*, np.int32_t*, int, int, int, int)
        int classify(np.float32_t*)

cdef class knnGPU:
    cdef C_knnGPU* g

    cdef int data_points
    cdef int features
    cdef int neighbor_num
    cdef int label_num

    def __cinit__(self, np.ndarray[ndim=1, dtype=np.float32_t, mode="c"] data, np.ndarray[ndim=1, dtype=np.int32_t, mode="c"] labels, int neighbors, int points, int features, int label):

        self.data_points = points
        self.features = features
        self.neighbor_num = neighbors
        self.label_num = label

        # create class
        self.g = new C_knnGPU(&data[0], &labels[0], neighbors, points, features, label)

    def classify(self, np.ndarray[ndim=1, dtype=np.float32_t, mode="c"] point):
        return self.g.classify(&point[0])

