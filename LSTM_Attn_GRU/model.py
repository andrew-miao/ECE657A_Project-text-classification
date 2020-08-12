import torch
import torch.nn as nn
import torch.nn.functional as F
from LSTM_Attn_GRU.config_LSTMAttnGRU import Config


class LSTMAttnGRU(nn.Module):
    def __init__(self, output_size):
        super(LSTMAttnGRU, self).__init__()

        self.lstm = nn.LSTM(Config.embedding_size, Config.hidden_size, batch_first=True, num_layers=Config.n_layers)
        self.fc1 = nn.Linear(Config.embedding_size, Config.fc1_size)
        self.fc2 = nn.Linear(Config.fc1_size, Config.fc2_size)
        self.fc3 = nn.Linear(Config.gru_hidden_size, Config.fc3_size)
        self.fc4 = nn.Linear(Config.fc3_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.gru = nn.GRU(Config.embedding_size, Config.gru_hidden_size, batch_first=True)
        self.dropout = nn.Dropout()
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=2)

    def attention(self, x, lstm_out,):
        x_compress = self.fc1(x)
        attn_weight = self.softmax(self.fc2(x_compress))
        attn_out = torch.bmm(attn_weight, lstm_out)
        cat_out = torch.cat([x_compress, attn_out], dim=2)

        return cat_out

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        cat_out = self.attention(x, lstm_out)
        gru_out, _ = self.gru(cat_out)
        gru_out = gru_out[:, 0, :]
        output = self.dropout(self.fc3(gru_out))
        output = F.relu(self.fc4(output))
        output = self.logsoftmax(output)

        return output