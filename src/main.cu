#include "dataset.hh"
#include "nn.hh"
#include "./layers/linear_layer.hh"
#include "./layers/relu_layer.hh"
#include "./layers/sigmoid_layer.hh"

int main() {

	Dataset dataset(256, 50);
	BCELoss bce_loss;

	NN nn(0.01f);
	nn.addLayer(new LinearLayer(Shape(2, 1024)));
	nn.addLayer(new ReLU());
	nn.addLayer(new LinearLayer(Shape(1024, 1)));
	nn.addLayer(new Sigmoid());


	for (int i = 0; i < 1201; i++) {
		float epoch_loss = 0.0;

		for (int j = 0; j < 49; j++) {
			nn.forward(dataset.getBatches().at(j));
			nn.backward(dataset.getTargets().at(j));
			epoch_loss += bce_loss.computeLoss(dataset.getTargets().at(j), nn.getPred());
		}

		if (i% 100 == 0) {
			cout << i << " | Loss: " << epoch_loss /50 << std::endl;
		}
	}

	nn.forward(dataset.getBatches().at(49));
	nn.getPred().copyDeviceToHost();

	int correct_predictions = 0;

	Matrix Y_hat= nn.getPred();
	Matrix Y= dataset.getTargets().at(49);

	for (int i = 0; i < 256; i++) {
		float post_processed_prediction = Y_hat[i] > 0.5 ? 1 : 0;
		if (post_processed_prediction == Y[i]) {
			correct_predictions++;
		}
	}

	cout << correct_predictions << "/256 testing examples predicted correctly at a threshold of 0.5" << endl;

	return 0;
}
