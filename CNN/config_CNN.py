import torch.nn as nn

class Config(object):
    conv1_num_filter = 3
    conv1_size = 5
    conv2_num_filter = 3
    conv2_size = 3
    conv3_num_filter = 3
    conv3_size = 3
    maxpool_size = 2
    fc1_size = 135
    fc2_size = 32
    criterion = nn.CrossEntropyLoss()