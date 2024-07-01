#include "relu_layer.hh"

__global__ void reluActivationComputeOutput(float* Z, float* A, int Z_x, int Z_y){
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < Z_x * Z_y) {
		if (Z[i] > 0) {
			A[i] = Z[i];
		}
		else {
			A[i] = 0;
		}
	}
}

__global__ void reluActivationComputeError(float* Z, float* aErr, float* Err, int Z_x, int Z_y){
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < Z_x * Z_y) {
		if (Z[i] > 0) {
			Err[i] = aErr[i]; // aErr[i]* 1.0f;
		}
		else {
			Err[i] = 0;
		}
	}
}

ReLU::ReLU() {}

ReLU::~ReLU(){}

Matrix& ReLU::forward(Matrix& Z) {
	this->Z = Z;
	A.allocateMemoryIfNotAllocated(Z.shape);

	computeOutput(Z);

	return A;
}

void ReLU::computeOutput(Matrix& Z){
	dim3 block_size(256);
	dim3 grid_size((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);

	reluActivationComputeOutput<<<grid_size, block_size>>>(Z.device_data.get(), A.device_data.get(), Z.shape.x, Z.shape.y);

	cudaError_t err= cudaGetLastError();
	if(err != cudaSuccess){
		cerr<< " CUDA error in ReLU layer computeOutput: "<< cudaGetErrorString(err) << endl;
		exit(err);
	}
}

Matrix& ReLU::backward(Matrix& aErr) {
	Err.allocateMemoryIfNotAllocated(Z.shape);

	computeError(aErr);

	return Err;
}

void ReLU::computeError(Matrix& aErr){
	dim3 block_size(256);
	dim3 grid_size((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);
	reluActivationComputeError<<<grid_size, block_size>>>(Z.device_data.get(), aErr.device_data.get(), Err.device_data.get(), Z.shape.x, Z.shape.y);

	cudaError_t err= cudaGetLastError();
	if(err != cudaSuccess){
		cerr<< " CUDA error in ReLU layer computeError: "<< cudaGetErrorString(err) << endl;
		exit(err);
	}

}