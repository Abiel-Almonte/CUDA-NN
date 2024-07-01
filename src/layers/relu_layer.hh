#pragma once
#include "layer.hh"

class ReLU : public NNLayer {
private:
	Matrix Z;

	Matrix A; //=ReLU(Z)
	Matrix Err;//error (delta) used to backpropagate to the next layer(l-1)

	void computeOutput(Matrix& Z);
	void computeError(Matrix& aErr);

public:
	ReLU();
	~ReLU();

	Matrix& forward(Matrix& Z);
	Matrix& backward(Matrix& aErr);
};