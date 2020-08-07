import math
import time
import load_data
import torch
# import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.LSTM import LSTMClassifier
from models.RCNN import RCNN

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
    model.initHidden(device)
    for inputs, labels in valloader:
        inputs, labels = inputs.to(device, dtype=torch.long), labels.to(device, dtype=torch.long)
        output = model(inputs)
        loss = criterion(output, labels)
        predict = torch.argmax(output, 1)
        num_corrects = (predict.data == labels.data).float().sum()
        acc = 100.0 * num_corrects / batch_size
        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()

    return total_epoch_loss / len(valloader), total_epoch_acc / len(valloader)

def training(model, n_epcohes, criterion, optimizer, device, print_every=100):
    start = time.time()
    min_loss = np.inf
    for epoch in range(n_epcohes):
        running_loss = 0
        running_acc = 0
        model.to(device)
        steps = 0
        model.train()
        model.initHidden(device)
        for text, label in trainloader:
            optimizer.zero_grad()
            text, label = text.to(device, dtype=torch.long), label.to(device, dtype=torch.long)
            output = model(text)
            loss = criterion(output, label)
            predict = torch.argmax(output, 1)
            num_corrects = (predict.data == label.data).float().sum()
            acc = 100.0 * num_corrects / batch_size
            loss.backward()
            clip_gradient(model, 1e-1)
            optimizer.step()
            steps += 1
            running_loss += loss.item()
            running_acc += acc.item()
            if steps % 100 == 0:
                val_loss, val_acc = evalating(model, criterion, device)
                print('%d/%d (%s) train loss: %.3f, train accuracy: %.2f%%, test loss: %.3f, test accuracy: %.2f%%' %
                      (epoch, n_epcohes, timeSince(start), running_loss / 100, running_acc / 100, val_loss, val_acc))
                if val_loss < min_loss:
                    print('Validation loss decreased: %.4f -> %.4f' % (min_loss, val_loss))
                    min_loss = val_loss
                    print('Saving model...')
                    torch.save(model.state_dict(), './rcnn.pt')
                running_loss = 0.0
                running_acc = 0.0
                model.train()


n_epoches = 10
learning_rate = 0.01
batch_size = 400
output_size = 4
hidden_size = 256
embedding_length = 300
vocab_size, word_embeddings, trainloader, valloader, testloader = load_data.load_dataset()
model = RCNN(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('We use', device)
# loss_fn = F.cross_entropy
training(model, n_epoches, criterion, optimizer, device)
'''
for epoch in range(10):
    train_loss, train_acc = train_model(model, train_iter, epoch)
    val_loss, val_acc = eval_model(model, valid_iter)
    
    print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%')
    
test_loss, test_acc = eval_model(model, test_iter)
print(f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%')
'''

'''
# Let us now predict the sentiment on a single sentence just for the testing purpose.
test_sen1 = "This is one of the best creation of Nolan. I can say, it's his magnum opus. Loved the soundtrack and especially those creative dialogues."
test_sen2 = "Ohh, such a ridiculous movie. Not gonna recommend it to anyone. Complete waste of time and money."

test_sen1 = TEXT.preprocess(test_sen1)
test_sen1 = [[TEXT.vocab.stoi[x] for x in test_sen1]]

test_sen2 = TEXT.preprocess(test_sen2)
test_sen2 = [[TEXT.vocab.stoi[x] for x in test_sen2]]

test_sen = np.asarray(test_sen1)
test_sen = torch.LongTensor(test_sen)
test_tensor = Variable(test_sen, volatile=True)
test_tensor = test_tensor.cuda()
model.eval()
output = model(test_tensor, 1)
out = F.softmax(output, 1)
if (torch.argmax(out[0]) == 1):
    print ("Sentiment: Positive")
else:
    print ("Sentiment: Negative")
'''