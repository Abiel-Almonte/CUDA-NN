#pragma once
#include "shape.hh"
#include <memory>
#include <iostream>

using namespace std;

class Matrix {
private:
	bool device_allocated;
	bool host_allocated;

	void allocateCuda();
	void allocateHost();

public:

	Matrix(size_t x = 1, size_t y = 1);
	Matrix(Shape shape);

	Shape shape;

	std::shared_ptr<float> device_data;
	std::shared_ptr<float> host_data;


	void allocateMemory();
	void allocateMemoryIfNotAllocated(Shape shape);
	void copyHostToDevice();
	void copyDeviceToHost();


	float& operator[](const int index);
	const float& operator[](const int index) const;
};