import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, input_dim, num_classes, size_dim):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv2_drop = nn.Dropout2d(p=0.1)
        self.fc1 = nn.Linear(16 * size_dim, 120)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(120, 84)
        self.output = nn.Linear(84, num_classes)
        self.activation = nn.LogSoftmax(dim=1)
        self.size_dim = size_dim

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * self.size_dim)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.relu(x)
        x = self.output(x)
        x = self.activation(x)
        return x
 
class CNN_PACS(nn.Module):
    def __init__(self, input_dim, num_classes, size_dim):
        super(CNN_PACS, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 6, kernel_size=5, stride=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=3)
        self.conv2_drop = nn.Dropout2d(p=0.1)
        self.fc1 = nn.Linear(16 * size_dim, 120)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(120, 84)
        self.output = nn.Linear(84, num_classes)
        self.activation = nn.LogSoftmax(dim=1)
        self.size_dim = size_dim

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * self.size_dim)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.relu(x)
        x = self.output(x)
        x = self.activation(x)
        return x