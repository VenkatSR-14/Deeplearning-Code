"""
This is an example code for RNN
"""

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

# Setting up Hyperparameters
num_classes = 10
input_size = 28
sequence_length = 28
num_layers = 2
batch_size = 64
hidden_size = 256
num_epochs = 3
learning_rate = 0.005

# DataLoader and device settings
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

train_dataset = datasets.MNIST("dataset/", train = True, download = True, transform = transforms.ToTensor())
test_dataset = datasets.MNIST("dataset/", train = False, download = True, transform = transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)

#Class for RNN

class RNN(nn.Module):
    def __init__(self, num_layers, hidden_size, input_size, num_classes):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.rnn(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out

#Setting up the model
model = RNN(num_layers = num_layers, hidden_size = hidden_size, input_size = input_size, num_classes = num_classes)
model = model.to(device = device)

# Setting up the optimizers, loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

# Training Loop
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        data = data.to(device).squeeze(1)
        targets = targets.to(device)
        scores = model(data)
        loss = criterion(scores, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# check_accuracy
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).squeeze(1)
            y = y.to(device)
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
    model.train()
    print(num_correct, num_samples)
    return num_correct/num_samples

print(f"Accuracy on training set is {(check_accuracy(train_loader, model)) * 100 :.2f}")
print(f"Accuracy on training set is {(check_accuracy(test_loader, model)) * 100 :.2f}")