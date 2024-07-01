#include "dataset.hh"
#include <random>

//XOR Dataset
Dataset::Dataset(size_t batch_size, size_t number_of_batches) :
	batch_size(batch_size), number_of_batches(number_of_batches)
{
	for (int i= 0; i < number_of_batches; i++) {
		batches.push_back(Matrix(batch_size, 2));
		targets.push_back(Matrix(batch_size, 1));

		batches[i].allocateMemory();
		targets[i].allocateMemory();
		for (int j= 0; j < batch_size; j++) {
			float x= random_binary();
			float y= random_binary(); 
			batches[i][j]= x;
			batches[i][batch_size + j]= y;
			targets[i][j] = static_cast<float>(static_cast<int>(x) ^ static_cast<int>(y));
		}

		batches[i].copyHostToDevice();
		targets[i].copyHostToDevice();
	}
}


float Dataset::random_binary(){
	static default_random_engine randGen(random_device{}()); 
	static uniform_int_distribution<int>  uniform_dist(0, 1);
	return static_cast<float>(uniform_dist(randGen));
}


int Dataset::getNumOfBatches() {
	return number_of_batches;
}

std::vector<Matrix>& Dataset::getBatches() {
	return batches;
}

std::vector<Matrix>& Dataset::getTargets() {
	return targets;
}