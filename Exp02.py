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
from sklearn.metrics import f1_score

batch_size = 400

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

def training(model, n_epochs, criterion, optimizer, device, path, train_loader, val_loader, print_every=240):
    start = time.time()
    min_loss = np.inf
    for epoch in range(n_epochs):
        running_loss = 0
        running_acc = 0
        steps = 1
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

                running_loss = 0.0
                running_acc = 0.0
                model.train()

def computeF1Score(model, device, loader):
    f1 = 0.0
    model.eval()
    for inputs, labels in loader:
        inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device, dtype=torch.long)
        output = model(inputs)
        predict = torch.argmax(output, 1)
        f1 += f1_score(labels, predict, average='macro')

    return f1 / len(loader)



def computInputSize(H, W):
    H = math.floor((H - 6) / 2) + 1
    W = math.floor((W - 6) / 2) + 1
    H = math.floor((H - 4) / 2) + 1
    W = math.floor((W - 4) / 2) + 1
    H = math.floor((H - 4) / 2) + 1
    W = math.floor((W - 4) / 2) + 1

    return 3 * H * W


def experiment(dataset_name, data_size=1.0):
    n_epochs = 15
    learning_rate = 0.001
    if dataset_name == 'AGNews':
        output_size = 4
    elif dataset_name == 'Dbpedia':
        output_size = 14
    else:
        output_size = 2
    vocab_size, train_loader, val_loader, test_loader = load_data.load_dataset(dataset_name, data_size=data_size)
    input_size = computInputSize(300, 60)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Our device is', device)
    model_name = ['CNN', 'RCNN', 'LSTM_Attn', 'LSTM_Attn_GRU']
    cnn = CNN(input_size, output_size)
    rcnn = RCNN(output_size)
    lstm_attn = LSTMAttention(output_size)
    lstm_attn_gru = LSTMAttnGRU(output_size)
    model_list = [cnn, rcnn, lstm_attn, lstm_attn_gru]
    model_path = ['./CNN/cnn.pt', './RCNN/rcnn.pt', './LSTM_Attn/lstm_attn.pt', './LSTM_Attn_GRU/lstm_attn_gru.pt']
    counter = 0
    print_every = len(train_loader)
    for model, path in zip(model_list, model_path):
        print('Now we training %s' % (model_name[counter]))
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        training(model, n_epochs, criterion, optimizer, device, path, train_loader, val_loader, print_every)
        counter += 1

    device = torch.device('cpu')
    f1_list = []
    for model, path in zip(model_list, model_path):
        model = model.to(device)
        model.load_state_dict(torch.load(path))
        f1_list.append(computeF1Score(model, device, test_loader))

    return f1_list