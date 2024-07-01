#include "matrix.hh"

Matrix::Matrix(size_t x, size_t y):shape(x, y), device_data(nullptr), host_data(nullptr), device_allocated(false), host_allocated(false){}

Matrix::Matrix(Shape shape):Matrix(shape.x, shape.y){}

void Matrix::allocateHost() {

	if (!host_allocated) {
		host_data= shared_ptr<float>(new float[shape.x * shape.y],[&](float* ptr){delete[] ptr;});
		host_allocated= true;
	}

}

void Matrix::allocateCuda() {

	if (!device_allocated) {
		float* temp= nullptr;
		cudaError_t err= cudaMalloc(&temp, shape.x * shape.y * sizeof(float));

		if (err != cudaSuccess) {
			cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << endl;
			exit(err);
    	}

		device_data = shared_ptr<float>(temp, [&](float* ptr){cudaFree(ptr);});
		device_allocated = true;
	}

}


void Matrix::allocateMemory() {
	allocateCuda();
	allocateHost();
}

void Matrix::allocateMemoryIfNotAllocated(Shape shape) {

	if (!device_allocated && !host_allocated) {
		this->shape= shape;
		allocateMemory();
	}

}

void Matrix::copyDeviceToHost() {
	if (device_allocated && host_allocated) {
		cudaError_t err=  cudaMemcpy(host_data.get(), device_data.get(), shape.x * shape.y * sizeof(float), cudaMemcpyDeviceToHost);

		if (err != cudaSuccess) {
			cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << endl;
			exit(err);
    	}
	}
}
void Matrix::copyHostToDevice() {
	if (device_allocated && host_allocated) {
		cudaError_t err=  cudaMemcpy(device_data.get(), host_data.get(), shape.x * shape.y * sizeof(float), cudaMemcpyHostToDevice);

		if (err != cudaSuccess) {
			cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << endl;
			exit(err);
    	}
	}
}


float& Matrix::operator[](const int i) {
	return host_data.get()[i];
}

const float& Matrix::operator[](const int i) const {
	return host_data.get()[i];
}