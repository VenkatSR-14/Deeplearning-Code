# Imports
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F


#Hyperparameters and other parameters
in_channels = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 5
load_model = True


#Creating the network
class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = 8, kernel_size = (3,3), stride = (1,1), padding = (1,1))
        self.maxPool1 = nn.MaxPool2d(kernel_size = (2,2), stride = (2,2))
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = (3,3), stride = (1,1), padding = (1,1))
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxPool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxPool1(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x

train_dataset = datasets.MNIST(root = "datasets/", train = True, transform = transforms.ToTensor(),download = True)
train_loader = DataLoader(dataset = train_dataset, shuffle = True, batch_size = 64)
test_dataset = datasets.MNIST(root = "datasets/", train = False, transform = transforms.ToTensor(), download = True)
test_loader = DataLoader(dataset = test_dataset, shuffle = True, batch_size = 64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN(in_channels = in_channels, num_classes = num_classes).to(device = device)
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
criterion = nn.CrossEntropyLoss()

def save_checkpoint(checkpoint):
    print("Saving Checkpoint ==>")
    torch.save(checkpoint, "my_checkpoint.pth.tar")

def load_checkpoint(checkpoint):
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

if load_model:
    load_checkpoint(torch.load('my_checkpoint.pth.tar'))

# Training the network 
for epoch in range(num_epochs):
    losses = []
    if epoch % 3 == 0:
        checkpoint = {'state_dict':model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_checkpoint(checkpoint)

    for batch_idx, (data, targets) in enumerate(train_loader):
        print(f"Processing batch {batch_idx+1}/{len(train_loader)} in epoch {epoch}")
        data = data.to(device = device)
        targets = targets.to(device = device)
        scores = model(data)
        loss = criterion(scores, targets)
        losses.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Mean loss at the end of the epoch = {sum(losses)/len(losses)}")



    

# Checking output accuracy

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    for x,y in loader:
        x = x.to(device = device)
        y = y.to(device = device)
        scores = model(x)
        _, predictions = scores.max(1)
        num_correct += (predictions == y).sum()
        num_samples +=  predictions.size(0)
    print(f"Got {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}")
    model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)