#pragma once
#include "utils/matrix.hh"
#include <vector>

using namespace std;

class Dataset {
private:
	size_t batch_size;
	size_t number_of_batches;

	float random_binary();
	std::vector<Matrix> batches;
	std::vector<Matrix> targets;

public:

	Dataset(size_t batch_size, size_t number_of_batches);

	int getNumOfBatches();
	std::vector<Matrix>& getBatches();
	std::vector<Matrix>& getTargets();

};