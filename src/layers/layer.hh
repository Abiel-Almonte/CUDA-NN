#pragma once
#include "../utils/matrix.hh"

class NNLayer {

public:
	float learning_rate;
	virtual ~NNLayer() = 0;

	virtual Matrix& forward(Matrix& A) = 0;
	virtual Matrix& backward(Matrix& aErr) = 0;

};

inline NNLayer::~NNLayer(){}