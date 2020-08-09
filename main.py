import math
import time
import load_data
import torch
# import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
# from models.LSTM import LSTMClassifier
# from models.RCNN import RCNN
from RCNN.config_RCNN import Config
from RCNN.model import RCNN


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)

def evalating(model, criterion, device):
    total_epoch_loss = 0.0
    total_epoch_acc = 0.0
    model.eval()
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device, dtype=torch.long)
        output = model(inputs)
        loss = criterion(output, labels)
        predict = torch.argmax(output, 1)
        num_corrects = (predict.data == labels.data).float().sum()
        acc = 100.0 * num_corrects / batch_size
        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()

    return total_epoch_loss / len(val_loader), total_epoch_acc / len(val_loader)

def training(model, n_epcohes, criterion, optimizer, device, print_every=200):
    start = time.time()
    min_loss = np.inf
    for epoch in range(n_epcohes):
        running_loss = 0
        running_acc = 0
        model.to(device)
        steps = 0
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
            # clip_gradient(model, 1e-1)
            optimizer.step()
            steps += 1
            running_loss += loss.item()
            running_acc += acc.item()
            if steps % print_every == 0:
                val_loss, val_acc = evalating(model, criterion, device)
                print('%d/%d (%s) train loss: %.3f, train accuracy: %.2f%%, test loss: %.3f, test accuracy: %.2f%%' %
                      (epoch, n_epcohes, timeSince(start), running_loss / print_every, running_acc / print_every, val_loss, val_acc))
                if val_loss < min_loss:
                    print('Validation loss decreased: %.4f -> %.4f' % (min_loss, val_loss))
                    min_loss = val_loss
                    print('Saving model...')
                    torch.save(model.state_dict(), './rcnn.pt')
                running_loss = 0.0
                running_acc = 0.0
                model.train()



n_epochs = 10
learning_rate = 0.0001
batch_size = 400
output_size = 4
vocab_size, train_loader, val_loader, test_loader = load_data.load_dataset()
model = RCNN(Config, output_size)
criterion = Config.criterion
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('We use', device)
training(model, n_epochs, criterion, optimizer, device)