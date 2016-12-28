#include <cuda_runtime.h>

class knnGPU
{
public:
	knnGPU(float* data_in, int* labels_in, int neighbors, int points, int features, int label);

	~knnGPU();

	void Cleanup();

	int BLK(int num, int blksize) {
		return (num + blksize - 1) / blksize;	
	}

    	int classify(float* point_in);

	int 			npoints;
	int 			nfeatures;
	int 			k;
	int	                nlabels;

	float 			*data;
	int 			*labels;
        int                     *sortedLabels;
	float			*distances;

	// size
	size_t data_bytes;
        size_t distance_bytes;
        size_t label_bytes;
	
	// kernel configuration
	int blocksize;

	dim3 blkDim;
	dim3 grdDim;
};
