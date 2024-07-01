#include "layer.hh"

class LinearLayer : public NNLayer {
private:
	const float weights_init_threshold = 0.01;
	float golort_init_std; 

	Matrix W; //Weight Matrix
	Matrix A; // input/ previous output vector
	Matrix b; //bias vector

	Matrix Z; // = W@A + b
	Matrix Err; // error (delta) used to backpropagate to the next layer(l-1)

	void initBias();
	void initWeights();

	void computeError(Matrix& dZ);
	void computeOutput(Matrix& A);
	void updateWeights(Matrix& dZ);
	void updateBias(Matrix& );

public:
	LinearLayer(Shape shape);
	~LinearLayer();

	Matrix& forward(Matrix& A);
	Matrix& backward(Matrix& dZ);

};