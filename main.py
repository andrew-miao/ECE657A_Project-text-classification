import math
import time
import load_data
import torch
# import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
from RCNN.config_RCNN import Config
from RCNN.model import RCNN
# from CNN.config_CNN import Config
# from CNN.model import CNN


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

def evalating(model, criterion, device ,loader):
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

def training(model, n_epcohes, criterion, optimizer, device, print_every=200):
    start = time.time()
    min_loss = np.inf
    for epoch in range(n_epcohes):
        running_loss = 0
        running_acc = 0
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
                val_loss, val_acc = evalating(model, criterion, device, val_loader)
                print('%d/%d (%s) train loss: %.3f, train accuracy: %.2f%%, val loss: %.3f, val accuracy: %.2f%%' %
                      (epoch, n_epcohes, timeSince(start), running_loss / print_every, running_acc / print_every, val_loss, val_acc))
                if val_loss < min_loss:
                    print('Validation loss decreased: %.4f -> %.4f' % (min_loss, val_loss))
                    min_loss = val_loss
                    print('Saving model...')
                    # torch.save(model.state_dict(), './cnn.pt')
                    torch.save(model.state_dict(), './rcnn.pt')
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


n_epochs = 15
learning_rate = 0.001
batch_size = 400
output_size = 4
vocab_size, train_loader, val_loader, test_loader = load_data.load_dataset()
input_size = computInputSize(300, 60)
criterion = Config.criterion
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = CNN(Config, input_size, output_size).to(device)
model = RCNN(Config, output_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
print('We use', device)
training(model, n_epochs, criterion, optimizer, device)
model.load_state_dict(torch.load('./rcnn.pt'))
test_loss, test_acc = evalating(model, criterion, device, test_loader)
print('test loss: %.3f, test accuracy: %.2f%%' % (test_loss, test_acc))