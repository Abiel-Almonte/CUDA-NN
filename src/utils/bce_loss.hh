#pragma once
#include "matrix.hh"

class BCELoss {
public:
	float computeLoss(Matrix target, Matrix pred);
	Matrix computeError(Matrix target, Matrix pred, Matrix Err);
};
