#include "bce_loss.hh"
#include "assert.h"


__global__ void binaryCrossEntropyLoss(float* pred, float* target, int pred_x, float* loss) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < pred_x) {
		float partial_cost = target[i] * logf(pred[i]) + (1.0f - target[i]) * logf(1.0f - pred[i]);
		atomicAdd(loss, - partial_cost / static_cast<float>(pred_x));
	}
}

__global__ void binaryCrossEntropyError(float* pred, float* target, float* Err, int pred_x) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < pred_x) {
		Err[i] = -1.0 * ( target[i]/pred[i] - (1 - target[i])/(1 - pred[i]) );
	}
}

float BCELoss::computeLoss(Matrix target, Matrix pred) {
	assert(pred.shape.x == target.shape.x);

	float* loss;
	cudaError_t err= cudaMallocManaged(&loss, sizeof(float));

	if (err != cudaSuccess){
		cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << endl;
	}

	*loss = 0.0f;

	dim3 block_size(256);
	dim3 num_of_blocks((pred.shape.x + block_size.x - 1) / block_size.x);
	binaryCrossEntropyLoss<<<num_of_blocks, block_size>>>(pred.device_data.get(), target.device_data.get(), pred.shape.x, loss);

	err= cudaGetLastError();
	if(err != cudaSuccess){
		cerr << "BCE computeLoss failed: " << cudaGetErrorString(err) << endl;
	}

	cudaDeviceSynchronize();

	float loss_value = *loss;
	cudaFree(loss);

	return loss_value;
}

Matrix BCELoss::computeError(Matrix target, Matrix pred, Matrix Err) {
	assert(pred.shape.x == target.shape.x);

	dim3 block_size(256);
	dim3 num_of_blocks((pred.shape.x + block_size.x - 1) / block_size.x);
	binaryCrossEntropyError<<<num_of_blocks, block_size>>>(pred.device_data.get(), target.device_data.get(), Err.device_data.get(), pred.shape.x);

	cudaError_t err= cudaGetLastError();
	if(err != cudaSuccess){
		cerr << "BCE computeError failed: " << cudaGetErrorString(err) << endl;
	}

	return Err;
}