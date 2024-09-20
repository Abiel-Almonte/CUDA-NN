#include <torch/torch.h>
#include <unistd.h>
#include <iostream>
#include <chrono>

class CustomDataset: public torch::data::datasets::Dataset<CustomDataset>{
    const int batch_size_;
    const int steps_;
    const torch::Tensor data, targets;

    public:
        CustomDataset(int batch_size, int steps): 
            batch_size_(batch_size),
            steps_(steps),
            data(torch::randn({steps_ * batch_size_, 2}, torch::device(torch::kCUDA))),
            targets((torch::sigmoid(data.sum(1, true))> 0.5).to(torch::kFloat32)){}

        torch::optional<size_t> size() const override{
            return steps_;
        }

        torch::data::Example<> get(size_t index) override{
            auto batch_start= index * batch_size_;
            auto batch_end= batch_start + batch_size_;
            auto sliced_data= data.slice(0, batch_start, batch_end);
            auto sliced_targets= targets.slice(0, batch_start, batch_end);
            return {sliced_data, sliced_targets};
        }

};

struct NN: torch::nn::Module{
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    torch::nn::ReLU relu{nullptr};
    torch::nn::Sigmoid sigmoid{nullptr};

    NN(){
        fc1= register_module("fc1", torch::nn::Linear(2, 50));
        relu= register_module("relu", torch::nn::ReLU());
        fc2= register_module("fc2", torch::nn::Linear(50, 1));
        sigmoid= register_module("sigmoid", torch::nn::Sigmoid());
    }

    torch::Tensor forward(torch::Tensor x){
        x= fc1->forward(x);
        x= relu->forward(x);
        x= fc2->forward(x);
        return sigmoid->forward(x);
    }

    torch::Tensor operator() (torch::Tensor x){
        return this->forward(x);
    }

};

float compute_accuracy(torch::Tensor predictions, torch::Tensor targets){
    auto pred_classes= (predictions> 0.5).to(torch::kFloat32);
    auto correct = pred_classes.eq(targets).sum().item<float>();
    return correct/ targets.size(0);
}

int main(int argc, char* argv[]){


    torch::manual_seed(42);
    torch::cuda::manual_seed(42);

    int epochs= 100;
    int steps= 400;
    int batch_size= 64; 

    int opt;
    while((opt = getopt(argc, argv, "e:s:b:")) != -1){
        switch(opt){
            case 'e':
                epochs= std::atoi(optarg);
                break;
            case 's':
                steps= std::atoi(optarg);
                break;
            case 'b':
                batch_size= std::atoi(optarg);
                break;
             default:
                std::cerr << "Usage: " << argv[0] << " [-e epochs] [-s steps] [-b batch_size]" << std::endl;
                return 1;
        }
    }

    float learning_rate= 0.01f;

    std::cout << "Epochs: " << epochs << "\n"
            << "Steps: " << steps << "\n"
            << "Batch Size: " << batch_size << std::endl;
        
    CustomDataset dataset(batch_size, steps);
    NN model;
    model.to(torch::kCUDA);

    auto bce_loss= torch::nn::BCELoss();
    torch::optim::SGD optimizer(model.parameters(), torch::optim::SGDOptions(learning_rate));

    torch::cuda::synchronize();
    auto start_time = std::chrono::high_resolution_clock::now();

    for(int i= 1; i<= epochs; i++){
        float epoch_loss= 0.0f;

        for (int batch_idx= 0; batch_idx < dataset.size().value()-1;  batch_idx++){
            auto batch= dataset.get(batch_idx);

            auto data = batch.data.to(torch::kCUDA);
            auto targets = batch.target.to(torch::kCUDA);

            auto predictions= model(data);
            auto loss= bce_loss(predictions, targets);

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            epoch_loss += loss.item<float>();
        }

        if (i %100 == 0){
            std::cout << i << " | Loss: " << epoch_loss / steps << std::endl;
        }
    }

    torch::cuda::synchronize();
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration =  std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    std::cout << "Time taken: " << duration / 1000.0 << " seconds\n";

    {
        torch::NoGradGuard no_grad;
        auto last_batch= dataset.get(steps- 1);
        auto data= last_batch.data.to(torch::kCUDA);
        auto targets= last_batch.target.to(torch::kCUDA);

        auto predictions= model(data);
        auto accuracy= compute_accuracy(predictions, targets);
        std::cout << "Testing accuracy: " << accuracy * 100 << "%\n";
    }

    return 0; 
}