import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, config, input_size, output_size):
        super(CNN, self).__init__()
        self.input_size = input_size

        self.conv1 = nn.Conv2d(1, config.conv1_num_filter, config.conv1_size)
        self.conv2 = nn.Conv2d(config.conv1_num_filter, config.conv2_num_filter, config.conv2_size)
        self.conv3 = nn.Conv2d(config.conv2_num_filter, config.conv3_num_filter, config.conv3_size)
        self.maxpool = nn.MaxPool2d(config.maxpool_size, config.maxpool_size)
        self.fc1 = nn.Linear(input_size, config.fc1_size)
        self.fc2 = nn.Linear(config.fc1_size, config.fc2_size)
        self.fc3 = nn.Linear(config.fc2_size, output_size)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))
        x = self.maxpool(F.relu(self.conv3(x)))
        out = x.view(-1, self.input_size)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out