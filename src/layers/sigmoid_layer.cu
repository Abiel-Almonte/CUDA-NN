#include "sigmoid_layer.hh"

__device__ float sigmoid_func(float x) {
	return 1.0f / (1 + exp(-x));
}

__global__ void sigmoidActivationComputeOutput(float* Z, float* A, int Z_x, int Z_y) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < Z_x * Z_y) {
		A[i] = sigmoid_func(Z[i]);
	}
}

__global__ void sigmoidActivationComputeError(float* Z, float*aErr, float* Err, int Z_x, int Z_y) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < Z_x * Z_y) {
		Err[i] = aErr[i] * sigmoid_func(Z[i]) * (1 - sigmoid_func(Z[i]));
	}
}

Sigmoid::Sigmoid() {}

Sigmoid::~Sigmoid(){}

Matrix& Sigmoid::forward(Matrix& Z) {
	this->Z = Z;
	A.allocateMemoryIfNotAllocated(Z.shape);

	computeOutput(Z);

	return A;
}

void Sigmoid::computeOutput(Matrix& Z){
	dim3 block_size(256);
	dim3 grid_size((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);
	sigmoidActivationComputeOutput<<<grid_size, block_size>>>(Z.device_data.get(), A.device_data.get(), Z.shape.x, Z.shape.y);

	cudaError_t err= cudaGetLastError();
	if(err != cudaSuccess){
		cerr<<"CUDA error in Sigmoid layer computeOutput: " <<cudaGetErrorString(err) <<endl;
	}
}

Matrix& Sigmoid::backward(Matrix& aErr) {
	Err.allocateMemoryIfNotAllocated(Z.shape);

	computeError(aErr);

	return Err;
}

void Sigmoid::computeError(Matrix& aErr){
	dim3 block_size(256);
	dim3 grid_size((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);
	sigmoidActivationComputeError<<<grid_size, block_size>>>(Z.device_data.get(), aErr.device_data.get(), Err.device_data.get(), Z.shape.x, Z.shape.y);
	
	cudaError_t err= cudaGetLastError();
	if(err != cudaSuccess){
		cerr<<"CUDA error in Sigmoid layer computeError: " <<cudaGetErrorString(err) <<endl;
	}
}
