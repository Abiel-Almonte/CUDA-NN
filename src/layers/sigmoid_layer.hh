#pragma once
#include "layer.hh"

class Sigmoid : public NNLayer {
private:
	Matrix A;

	Matrix Z;
	Matrix Err;

	void computeOutput(Matrix& Z);
	void computeError(Matrix& aErr);

public:
	Sigmoid();
	~Sigmoid();

	Matrix& forward(Matrix& Z);
	Matrix& backward(Matrix& aErr);
};