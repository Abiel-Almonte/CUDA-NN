#pragma once
#include "layers/layer.hh"
#include "utils/bce_loss.hh"
#include <vector>

class NN{
private:
	std::vector<NNLayer*> layers;
	BCELoss bce_loss;

	Matrix Y;
	Matrix dY;
	float learning_rate;

public:
	NN(float learning_rate = 0.01);
	~NN();

	Matrix forward(Matrix& X);
	void backward(Matrix& target);

	void addLayer(NNLayer *layer);
	Matrix getPred();
	std::vector<NNLayer*> getLayers() const;

};