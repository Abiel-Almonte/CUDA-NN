import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import time

class Dataset(torch.utils.data.Dataset):
    def __init__(self, batch_size, steps):
        super(Dataset, self).__init__()
        self.batch_size = batch_size
        self.steps = steps
        self.data = torch.randn(steps * batch_size, 2).to('cuda')
        self.targets = (torch.sigmoid(self.data.sum(dim=1, keepdim=True)) > 0.5).float()

    def __len__(self):
        return self.steps

    def __getitem__(self, index):
        batch_start = index * self.batch_size
        batch_end = batch_start + self.batch_size
        return self.data[batch_start:batch_end], self.targets[batch_start:batch_end]
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.linear1 = nn.Linear(2, 50)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(50, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return self.sigmoid(x)
    
def compute_accuracy(predictions, targets):
    pred_classes = (predictions > 0.5).float()
    correct = (pred_classes == targets).sum().item()
    return correct / len(targets)

def parse_args():
    parser = argparse.ArgumentParser(description='Training configuration.')
    
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training.')
    parser.add_argument('--steps', type=int, default=400, help='Number of steps per epoch.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training.')
    
    return parser.parse_args()

if __name__ == "__main__":
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    args = parse_args()
    
    epochs = args.epochs
    steps = args.steps
    batch_size = args.batch_size
    
    print(f"Epochs: {epochs}")
    print(f"Steps: {steps}")
    print(f"Batch Size: {batch_size}")
    learning_rate = 0.01

    dataset = Dataset(batch_size, steps)
    model = NN().to('cuda')
    bce_loss = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    torch.cuda.synchronize()
    start_time= time.perf_counter()
    for epoch in range(1, epochs):
        epoch_loss = 0.0
        for batch_idx in range(len(dataset)-1):
            data, targets = dataset[batch_idx]
            predictions = model(data)
            loss = bce_loss(predictions, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        if epoch % 100 == 0:
            print(f'{epoch} | Loss: {epoch_loss / steps}')
    torch.cuda.synchronize()
    end_time= time.perf_counter()

    print(f'Time taken: {end_time - start_time:.5f} seconds')
    with torch.no_grad():
        data, targets = dataset[steps-1]
        predictions = model(data)

        accuracy = compute_accuracy(predictions, targets)
        print(f'Testing accuracy: {accuracy * 100:.2f}%')