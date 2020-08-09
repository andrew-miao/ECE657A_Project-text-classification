import torch.nn as nn

class Config(object):
    embedding_size = 300
    n_layers = 1
    hidden_size = 64
    drop_prob = 0.2
    criterion = nn.NLLLoss()