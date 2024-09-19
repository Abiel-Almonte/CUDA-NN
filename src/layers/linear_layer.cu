#include "linear_layer.hh"
#include <random>
#include <assert.h>

#define BLOCKDIM 32
#define FULL_MASK 0xffffffff

__global__ void linearLayerComputeOutput(const float* __restrict__ W, const float* __restrict__ A, float* __restrict__ Z, const float* __restrict__ b,int W_x, int W_y, int A_x, int A_y){
	__shared__ float tileW[BLOCKDIM][BLOCKDIM];
	__shared__ float tileA[BLOCKDIM][BLOCKDIM];

	const int tx = threadIdx.x;
	const int ty = threadIdx.y;

	const int col= blockIdx.x * BLOCKDIM + tx;
	const int row= blockIdx.y * BLOCKDIM + ty;

	const int Z_x= A_x;
	const int Z_y= W_y;

	float element_value= 0.0f;

	for (int i= 0; i < ((W_x + BLOCKDIM -1) / BLOCKDIM); ++i) {
		if (tx + i*BLOCKDIM < W_x && row < W_y){
			tileW[ty][tx] = W[row*W_x + i*BLOCKDIM + tx];
		}
		else{
			tileW[ty][tx]= 0.0f;
		}

		if (ty + i*BLOCKDIM < A_y && col < A_x){
			tileA[ty][tx] = A[(ty + i *BLOCKDIM) * A_x + col];
		}
		else{
			tileA[ty][tx]= 0.0f;
		}
		
		__syncthreads();

		#pragma unroll
		for (int k = 0; k < BLOCKDIM; k++) {
			element_value += tileW[ty][k] * tileA[k][tx];
		}

		__syncthreads();
	}

	if (row < Z_y && col < Z_x) {
        Z[row * Z_x + col] = element_value + b[row];
    }
}

__global__ void linearLayerComputeError(const float* __restrict__ W, const float* __restrict__ aErr, float* __restrict__ Err, int W_x, int W_y, int aErr_x, int aErr_y){
	__shared__ float tileW[BLOCKDIM][BLOCKDIM];
	__shared__ float tileaErr[BLOCKDIM][BLOCKDIM];

	const int col= blockIdx.x * BLOCKDIM + threadIdx.x;
	const int row= blockIdx.y * BLOCKDIM + threadIdx.y;

	const int tx= threadIdx.x;
	const int ty= threadIdx.y;

	const int Err_x= aErr_x;
	const int Err_y= W_x;

	float element_value= 0.0f;

	for (int i= 0; i < ((W_y + BLOCKDIM -1)/ BLOCKDIM) ; i++) {

		if(tx + i*BLOCKDIM< W_y && row < W_x){
			tileW[ty][tx]= W[(tx + i*BLOCKDIM) * W_x + row];
		}
		else{
			tileW[ty][tx]= 0.0f;
		}

		if(ty + i*BLOCKDIM < aErr_y && col < aErr_x){
			tileaErr[ty][tx]= aErr[(ty + i*BLOCKDIM) * aErr_x + col];
		}
		else{
			tileW[ty][tx]= 0.0f;
		}

		
		__syncthreads();

		#pragma unroll
		for(int k = 0; k < BLOCKDIM; k ++){
			element_value += tileW[k][ty] * tileaErr[k][tx];
		}

		__syncthreads();
	}

	if (row < Err_y && col < Err_x) {
		Err[row * Err_x + col]= element_value;
	}
}

__inline__ __device__ float warpReduceSum(float val){
	unsigned mask= __activemask();
	for (int offset= BLOCKDIM;  offset>0; offset/=2){
		val += __shfl_down_sync(mask, val, offset);
	}
	return val;
}

__global__ void linearLayerUpdateWeights(const float* __restrict__ aErr, const float* __restrict__ A, float* __restrict__ W, int aErr_x, int aErr_y, int A_x, int A_y, const float learning_rate){
	__shared__ float tileA[BLOCKDIM][BLOCKDIM];
	__shared__ float tileaErr[BLOCKDIM][BLOCKDIM];

	const int tx= threadIdx.x;
	const int ty= threadIdx.y; 

	const int col= blockIdx.x *BLOCKDIM +tx;
	const int row= blockIdx.y * BLOCKDIM + ty;

	const int W_x= A_y;
	const int W_y= aErr_y;

	float element_value= 0.0f;

	for(int i =0; i < (aErr_x+ BLOCKDIM -1)/ BLOCKDIM; i++){
		if(tx + i*BLOCKDIM < aErr_x && row < aErr_y){
			tileaErr[ty][tx]= aErr[row*aErr_x + tx + i*BLOCKDIM];
		}
		else{
			tileaErr[ty][tx]= 0.0f;
		}
		if(ty + i*BLOCKDIM < A_x &&  col < A_y){
			tileA[ty][tx]= A[col * A_x + i * BLOCKDIM + ty];
		}
		else{
			tileA[ty][tx]= 0.0f;
		}

		__syncthreads();

		#pragma unroll
		for(int k = 0; k < BLOCKDIM; k ++){
			element_value += tileaErr[ty][k] * tileA[k][tx];
		}

		__syncthreads();
	}

	if (row < W_y && col < W_x) {
        atomicAdd(&W[row * W_x + col], -learning_rate * element_value);
	}
}

__global__ void linearLayerUpdateBias(const float* __restrict__ aErr, float* __restrict__ b, int aErr_x, int aErr_y,int b_x, const float learning_rate){
	__shared__ float data[8];

	const int tx= threadIdx.x;
	const int bx= blockIdx.x;
	const int idx= bx * BLOCKDIM+ tx;

	const int lane= tx % 32;
	const int wid= tx / 32;

	float sum= 0.0f;

	for (int i= idx; i < aErr_x * aErr_y; i += gridDim.x *blockDim.x) sum += aErr[i];

	sum= warpReduceSum(sum);

	if(lane==0) data[wid]= sum;

	__syncthreads();

	if(tx< 8){
		sum= data[tx];
		sum= warpReduceSum(sum);

		if(lane==0) atomicAdd(&b[bx], -learning_rate * sum);
	}
}

LinearLayer::LinearLayer(Shape shape):W(shape), b(shape.y, 1), golort_init_std(sqrt(2.0f/ static_cast<float>(shape.x + shape.y))){

	b.allocateMemory();
	W.allocateMemory();

	initBias();
	initWeights();
	
	b.copyHostToDevice();
	W.copyHostToDevice();
}

LinearLayer::~LinearLayer(){}

void LinearLayer::initWeights(){
	default_random_engine randGen(random_device{}());
	normal_distribution<float> normal_dist(0.0f, golort_init_std);

	for (int row = 0; row < W.shape.y; row++) {
		for (int col = 0; col < W.shape.x; col++) {
			W[row* W.shape.x + col] = normal_dist(randGen) * weights_init_threshold;
		}
	}
}

void LinearLayer::initBias() {
	for (int i = 0; i < b.shape.x; i++){
		b[i] = 0.0f;
	}
}

Matrix& LinearLayer::forward(Matrix& A){
	assert(W.shape.x == A.shape.y);

	this->A = A;
	Shape Z_shape(A.shape.x, W.shape.y);
	Z.allocateMemoryIfNotAllocated(Z_shape);

	computeOutput(A);

	return Z;
}

void LinearLayer::computeOutput(Matrix& A){
	dim3 block_size(BLOCKDIM, BLOCKDIM);
	dim3 grid_size((Z.shape.x + BLOCKDIM- 1) / BLOCKDIM, (Z.shape.y + BLOCKDIM - 1) / BLOCKDIM);
	linearLayerComputeOutput<<<grid_size, block_size>>>( W.device_data.get(), A.device_data.get(),Z.device_data.get(), b.device_data.get(), W.shape.x, W.shape.y, A.shape.x, A.shape.y);

	cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in linear layer computeOutput: " << cudaGetErrorString(err) << std::endl;
        exit(err);
    }
}

Matrix& LinearLayer::backward(Matrix& aErr){
	assert(aErr.shape.y == W.shape.y);
	Err.allocateMemoryIfNotAllocated(A.shape);

	computeError(aErr);
	updateBias(aErr);
	updateWeights(aErr);

	return Err;
}

void LinearLayer::computeError(Matrix& aErr) {
	dim3 block_size(BLOCKDIM, BLOCKDIM);
	dim3 grid_size((A.shape.x + BLOCKDIM - 1) / BLOCKDIM, (A.shape.y + BLOCKDIM - 1) / BLOCKDIM);
	linearLayerComputeError<<<grid_size, block_size>>>( W.device_data.get(), aErr.device_data.get(), Err.device_data.get(), W.shape.x, W.shape.y, aErr.shape.x, aErr.shape.y);

	cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in linear layer computeError: " << cudaGetErrorString(err) << std::endl;
        exit(err);
    }
}

void LinearLayer::updateWeights(Matrix& aErr) {
	dim3 block_size(16, 16);
	dim3 grid_size((W.shape.x + BLOCKDIM - 1) / BLOCKDIM , (W.shape.y + BLOCKDIM - 1) / BLOCKDIM);
	linearLayerUpdateWeights<<<grid_size, block_size>>>(aErr.device_data.get(), A.device_data.get(), W.device_data.get(), aErr.shape.x, aErr.shape.y, A.shape.x, A.shape.y, learning_rate);
	
	cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in linear layer updateWeights: " << cudaGetErrorString(err) << std::endl;
        exit(err);
    }
}

void LinearLayer::updateBias(Matrix& aErr) {
	dim3 block_size(256);
	dim3 grid_size((aErr.shape.y * aErr.shape.x + BLOCKDIM- 1) / BLOCKDIM);
	linearLayerUpdateBias<<<grid_size, block_size>>>(aErr.device_data.get(), b.device_data.get(), aErr.shape.x, aErr.shape.y, b.shape.x, learning_rate);

	cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in updateWeights: " << cudaGetErrorString(err) << std::endl;
        exit(err);
    }
}