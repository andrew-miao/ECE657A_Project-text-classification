import math
import time
import load_data
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from RCNN.model import RCNN
from CNN.model import CNN
from LSTM_Attn.model import LSTMAttention
from LSTM_Attn_GRU.model import LSTMAttnGRU
import sys
from torch.utils.tensorboard import SummaryWriter

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def evalating(model, criterion, device, loader):
    total_epoch_loss = 0.0
    total_epoch_acc = 0.0
    model.eval()
    for inputs, labels in loader:
        inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device, dtype=torch.long)
        output = model(inputs)
        loss = criterion(output, labels)
        predict = torch.argmax(output, 1)
        num_corrects = (predict.data == labels.data).float().sum()
        acc = 100.0 * num_corrects / batch_size
        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()

    return total_epoch_loss / len(loader), total_epoch_acc / len(loader)

def training(model, n_epochs, criterion, optimizer, device, path):
    start = time.time()
    min_loss = np.inf
    steps = 1
    print_every = len(train_loader)
    for epoch in range(n_epochs):
        running_loss = 0
        running_acc = 0
        model.train()
        for inputs, label in train_loader:
            optimizer.zero_grad()
            inputs, label = inputs.to(device, dtype=torch.float), label.to(device, dtype=torch.long)
            output = model(inputs)
            loss = criterion(output, label)
            predict = torch.argmax(output, 1)
            num_corrects = (predict.data == label.data).float().sum()
            acc = 100.0 * num_corrects / batch_size
            loss.backward()
            optimizer.step()
            steps += 1
            running_loss += loss.item()
            running_acc += acc.item()

            if steps % print_every == 0:
                val_loss, val_acc = evalating(model, criterion, device, val_loader)
                print('%d/%d (%s) train loss: %.3f, train accuracy: %.2f%%, val loss: %.3f, val accuracy: %.2f%%' %
                      (epoch + 1, n_epochs, timeSince(start), running_loss / print_every, running_acc / print_every, val_loss, val_acc))
                if val_loss < min_loss:
                    print('Validation loss decreased: %.4f -> %.4f' % (min_loss, val_loss))
                    min_loss = val_loss
                    print('Saving model...')
                    torch.save(model.state_dict(), path)

                writer.add_scalar('Train loss', running_loss / print_every, steps)
                writer.add_scalar('Val loss', val_loss, steps)
                writer.add_scalar('Train accuracy', running_acc / print_every, steps)
                writer.add_scalar('Val accuracy', val_acc, steps)
                running_loss = 0.0
                running_acc = 0.0
                model.train()



def computInputSize(H, W):
    H = math.floor((H - 6) / 2) + 1
    W = math.floor((W - 6) / 2) + 1
    H = math.floor((H - 4) / 2) + 1
    W = math.floor((W - 4) / 2) + 1
    H = math.floor((H - 4) / 2) + 1
    W = math.floor((W - 4) / 2) + 1

    return 3 * H * W


if len(sys.argv) != 3:
	sys.exit('Use: python main.py <model> <dataset>')

model_list = ['cnn', 'rcnn', 'lstm_attn', 'lstm_gru']
datasets = ['agnews', 'dbpedia', 'amazon', 'yelp']
model_name = sys.argv[1]
dataset = sys.argv[2]
if model_name not in model_list or dataset not in datasets:
    sys.exit('The model or dataset that we do not support now')

output_size_dict = {'agnews': 4, 'dbpedia': 14, 'amazon': 2, 'yelp': 2}
writer = SummaryWriter()
n_epochs = 15
learning_rate = 0.001
batch_size = 400
vocab_size, train_loader, val_loader, test_loader = load_data.load_dataset(dataset)
input_size = computInputSize(300, 60)
criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if model_name == 'cnn':
    model = CNN(input_size, output_size_dict[dataset]).to(device)
    path = './CNN/cnn.pt'
elif model_name == 'rcnn':
    model = RCNN(output_size_dict[dataset]).to(device)
    path = './RCNN/rcnn.pt'
elif model_name == 'lstm_attn':
    model = LSTMAttention(output_size_dict[dataset]).to(device)
    path = './LSTM_Attn/lstm_attn.pt'
elif model_name == 'lstm_gru':
    model = LSTMAttnGRU(output_size_dict[dataset]).to(device)
    path = './LSTM_Attn_GRU/lstm_attn_gru.pt'
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
print('We use', device)
training(model, n_epochs, criterion, optimizer, device, path)
model.load_state_dict(torch.load(path))
writer.flush()
writer.close()
test_loss, test_acc = evalating(model, criterion, device, test_loader)
print('test loss: %.3f, test accuracy: %.2f%%' % (test_loss, test_acc))