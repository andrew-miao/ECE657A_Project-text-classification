import torch.nn as nn
import torch.nn.functional as F
from CNN.config_CNN import Config

class CNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(CNN, self).__init__()
        self.input_size = input_size

        self.conv1 = nn.Conv2d(1, Config.conv1_num_filter, Config.conv1_size)
        self.conv2 = nn.Conv2d(Config.conv1_num_filter, Config.conv2_num_filter, Config.conv2_size)
        self.conv3 = nn.Conv2d(Config.conv2_num_filter, Config.conv3_num_filter, Config.conv3_size)
        self.maxpool = nn.MaxPool2d(Config.maxpool_size, Config.maxpool_size)
        self.fc1 = nn.Linear(input_size, Config.fc1_size)
        self.fc2 = nn.Linear(Config.fc1_size, Config.fc2_size)
        self.fc3 = nn.Linear(Config.fc2_size, output_size)

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