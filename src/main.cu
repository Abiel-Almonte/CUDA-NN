#include <cuda_runtime.h>
#include <unistd.h>
#include "dataset.hh"
#include "nn.hh"
#include "./layers/linear_layer.hh"
#include "./layers/relu_layer.hh"
#include "./layers/sigmoid_layer.hh"

int main(int argc, char *argv[]) {
	int epochs= 100;
	int steps= 400;
	int batch_size= 64;

	int opt;
    while ((opt = getopt(argc, argv, "e:s:b:")) != -1) {
        switch (opt) {
            case 'e':
                epochs = std::atoi(optarg);
                break;
            case 's':
                steps = std::atoi(optarg);
                break;
            case 'b':
                batch_size = std::atoi(optarg);
                break;
            default:
                std::cerr << "Usage: " << argv[0] << " [-e epochs] [-s steps] [-b batch_size]" << std::endl;
                return 1;
        }
    }

	std::cout << "Epochs: " << epochs << std::endl;
    std::cout << "Steps: " << steps << std::endl;
    std::cout << "Batch Size: " << batch_size << std::endl;

	Dataset dataset(batch_size, steps);
	BCELoss bce_loss;

	NN nn(0.01f);
	nn.addLayer(new LinearLayer(Shape(2, 50)));
	nn.addLayer(new ReLU());
	nn.addLayer(new LinearLayer(Shape(50, 1)));
	nn.addLayer(new Sigmoid());

	cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
	float final_loss;
	for (int i = 0; i < epochs; i++) {
		float epoch_loss = 0.0;

		for (int j = 0; j < steps-1; j++) {
			nn.forward(dataset.getBatches().at(j));
			nn.backward(dataset.getTargets().at(j));
			epoch_loss += bce_loss.computeLoss(dataset.getTargets().at(j), nn.getPred());
		}

		if (i% 100 == 0) {
			cout << i << " | Loss: " << epoch_loss /(steps-1) << std::endl;
		}

		final_loss= epoch_loss /(steps-1);
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout << "Final Loss for training: " << final_loss << std::endl;
    std::cout << "Time taken for training: " << milliseconds / 1000.0f << " seconds" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

	nn.forward(dataset.getBatches().at(steps-1));
	nn.getPred().copyDeviceToHost();

	int correct_predictions = 0;

	Matrix Y_hat= nn.getPred();
	Matrix Y= dataset.getTargets().at(steps-1);

	for (int i = 0; i < batch_size; i++) {
		float post_processed_prediction = Y_hat[i] > 0.5 ? 1 : 0;
		if (post_processed_prediction == Y[i]) {
			correct_predictions++;
		}
	}

	cout << correct_predictions << "/" <<batch_size <<" testing examples predicted correctly at a threshold of 0.5" << endl;

	return 0;
}
