#include "nn.hh"

NN::NN(float learning_rate):learning_rate(learning_rate){}

NN::~NN() {
	for (auto layer : layers) {
		delete layer;
	}
}

void NN::addLayer(NNLayer* layer) {
	layers.push_back(layer);
}

Matrix NN::forward(Matrix& X) {
	Matrix Z = X;

	for (const auto layer : layers) {
		Z = layer->forward(Z);
	}

	Y=Z;
	return Z;
}

void NN::backward(Matrix& target) {
	dY.allocateMemoryIfNotAllocated(Y.shape);
	Matrix error = bce_loss.computeError(target, Y, dY);

	for (auto it = this->layers.rbegin(); it != this->layers.rend(); it++) {
		(*it)->learning_rate= learning_rate;
		error = (*it)->backward(error);
	}

	cudaDeviceSynchronize();
}

std::vector<NNLayer*> NN::getLayers() const {
	return layers;
}

Matrix NN::getPred(){
	return Y;
}