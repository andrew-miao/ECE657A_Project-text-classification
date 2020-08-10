import torch
import torch.nn as nn
import torch.nn.functional as F
from LSTM_Attn_GRU.config_LSTMAttnGRU import Config

class LSTMAttnGRU(nn.Module):
    def __init__(self, output_size):
        super(LSTMAttnGRU, self).__init__()

        self.lstm = nn.LSTM(Config.embedding_size, Config.hidden_size, batch_first=True)
        self.fc1 = nn.Linear(2 * Config.hidden_size, Config.fc1_size)
        self.maxpool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.fc2 = nn.Linear(Config.hidden_size, Config.fc2_size)
        self.fc3 = nn.Linear(Config.fc2_size, Config.fc3_size)
        self.fc4 = nn.Linear(Config.fc3_size, output_size)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(Config.drop_prob)
        self.gru = nn.GRU(Config.hidden_size, Config.hidden_size, batch_first=True)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def attention(self, lstm_out, hidden_state, cell_state):
        hidden_state = hidden_state.squeeze(2)  # hidden_state.size() will change to (batch_size, hidden_size)
        cell_state = cell_state.squeeze(2)  # cell_state.size() will change to (batch_size, hidden_size)
        new_state = torch.cat([hidden_state, cell_state], dim=1)  # new_state.size() = (batch_size, 2 * hidden_size)
        new_state = self.fc1(new_state)  # new_state.size() = (batch_size, seq)
        attn_weights = self.softmax(new_state)
        attn_weights = attn_weights.unsqueeze(2)  # attn_weights.size() = (batch_size, seq, 1)
        attn_weights = attn_weights.permute(0, 2, 1)  # attn_weights.size() = (batch_size, 1, seq)
        new_output = torch.bmm(attn_weights, lstm_out)  # new_output.size() = (batch_size, 1, hidden)

        return new_output

    def forward(self, x):
        lstm_out, (hidden_state, cell_state) = self.lstm(x)
        attn_output = self.attention(lstm_out, hidden_state.permute(1, 2, 0), cell_state.permute(1, 2, 0))
        output, _ = self.gru(attn_output, hidden_state)
        output = output[:, 0, :]
        output = self.fc2(output)
        output = self.dropout(F.relu(output))
        output = self.fc3(output)
        output = self.dropout(F.relu(output))
        output = self.fc4(output)
        output = self.logsoftmax(output)

        return output