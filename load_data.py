# _*_ coding: utf-8 _*_

import torch
from torchtext import datasets
from torchtext.vocab import GloVe, Vocab
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

np.random.seed(42)
def splitFeaturesLabels(dataset):
    """
    dataset: dataset that we use
    return:
        features: a list of tensor (features)
        labels: a list of labels
    """
    features = [None] * len(dataset)
    labels = np.zeros((len(dataset),))
    for i in range(len(dataset)):
        (labels[i], features[i]) = dataset.__getitem__(i)

    return features, labels

def pad_input(data, pad_size=50):
    """
    data: the list of features
    pad_size: the size that user want to pad, which should be integer Default value = 50.
    return: a numpy array features with padding.
    """
    features = np.zeros((len(data), pad_size))
    for i in range(len(data)):
        tmp = data[i].numpy()
        features[i, -tmp.shape[0]:] = tmp[:pad_size]

    return features


def load_dataset(dataset_name=None):

    """
    dataset_name: the dataset that user want to use. Default is AGNews
    return:
        vocab_size: the number of vocabulary in the dataset.
        word_embeddings: word embedding that we used, which is a vector.
        train_loader: DataLoader for training set.
        val_loader: DataLoader for validation set.
        test_loader: DataLoader for testing set.
    """
    if not os.path.isdir('./.data'):
        os.mkdir('./.data')
    train_data = None
    batch_size = 400
    print("Loading dataset...")
    if dataset_name is None or dataset_name == 'AGNews':
        train_data, test_data = datasets.text_classification.DATASETS['AG_NEWS'](root='./.data')
    print("Loading word embedding pre-train model...")
    create_vocab = train_data.get_vocab()
    create_vocab = Vocab(create_vocab.freqs, vectors=GloVe(name='6B', dim=300))
    train_data, test_data = datasets.text_classification.AG_NEWS(vocab=create_vocab)
    print("Splitting and padding data...")
    train_data, train_labels = splitFeaturesLabels(train_data)
    test_data, y_test = splitFeaturesLabels(test_data)
    train_size = int(0.8 * len(train_data))
    np.random.shuffle(train_data)
    X_train, y_train = train_data[:train_size], train_labels[:train_size]
    X_val, y_val = train_data[train_size:], train_labels[train_size:]
    X_train = pad_input(X_train)
    X_val = pad_input(X_val)
    X_test = pad_input(test_data)
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    vocab_size = len(create_vocab)
    word_embeddings = create_vocab.vectors
    print("Finished data loading.")
    return vocab_size, word_embeddings, train_loader, val_loader, test_loader

if __name__ == '__main__':
    pass