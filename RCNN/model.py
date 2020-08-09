import torch
import torch.nn as nn
from torch.nn import functional as F


class RCNN(nn.Module):
    def __init__(self, config, output_size):
        super(RCNN, self).__init__()
        self.config = config
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size=self.config.embedding_size, hidden_size=self.config.hidden_size, num_layers=self.config.n_layers,
                            dropout=self.config.drop_prob, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(self.config.drop_prob)
        self.W = nn.Linear(self.config.embedding_size + 2 * self.config.hidden_size, self.config.hidden_size)
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(self.config.hidden_size, self.output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # lstm_out = (batch_size, seq_len, 2 * hidden_size)
        x = torch.cat([lstm_out, x], 2)
        linear_output = self.tanh(self.W(x))
        linear_output = linear_output.permute(0, 2, 1)  # linear_output = (batch_size, seq_len, hidden_size)
        max_out_features = F.max_pool1d(linear_output, linear_output.size()[2]).squeeze(2)
        max_out_features = self.dropout(max_out_features)
        out = self.fc(max_out_features)
        out = self.softmax(out)
        return out