# _*_ coding: utf-8 _*_

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import re
import pandas as pd
from collections import Counter
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')
np.random.seed(42)
stop_words = set(stopwords.words('english'))
stop_words.update('(', ')', ',', ';', '!', '.', '-', '\\', '--')

def pad_input(data, help_find_word, pad_size=60):
    """
    data: the list of words
    help_find_word: map known word to true, map unknown word to false
    pad_size: the size that user want to pad, which should be integer Default value = 50.
    return: a list of words with padding.
    """
    features = [None] * len(data)
    for i, sentence in enumerate(data):
        features[i] = ['pad'] * pad_size
        sentence = sentence[:pad_size]
        for j, word in enumerate(sentence):
            try:
                if help_find_word[word]:
                    features[i][j] = word
                else:
                    features[i][j] = 'unk'
            except KeyError:
                features[i][j] = 'unk'

    return features

def word_embed(data, embedding_dict, pad_size=60):
    """
    data: a list contains dataset, e.g. [[sentence1], [sentence2], ..., sentence[m]].
    embedding_dict: map a word to a 300d vector
    pad_size: the size of padding
    return:
        word_embeddings: a 3d array, which generated by Glove model
    """
    data_size = len(data)
    # word_embeddings = np.zeros((data_size, pad_size, 300))
    word_embeddings = torch.zeros(data_size, pad_size, 300)
    for i, sentence in enumerate(data):
        for j, word in enumerate(sentence):
            word_embeddings[i][j] = embedding_dict[word]

    return word_embeddings

def preprocessing_train_data(train, dataset_name):
    """
    train: the dataset of training data, and the data type is pandas dataframe.
    dataset_name: the name of dataset that user use, and the data type is string.
    return:
        train_data: a 3d array (num_document, pad_sentence_size, embedding_size).
        train_labels: an array that contains the true labels in training dataset.
        help_find_word: map known word to true, map unknown word to false.
        embedding_dict: map a word to a 300d vector.
    """
    train[0] = train[0] - 1
    train_labels = train[0].to_numpy()
    train['pad'] = ' '
    if dataset_name != 'yelp_review_polarity_csv':
        train['sentence'] = train[1] + train['pad'] + train[2]
    else:
        train['sentence'] = train[1]
    sentences = train['sentence']
    words = Counter()
    train_sentences = [None] * len(sentences)
    help_find_word = {}
    for i, sentence in enumerate(sentences):
        train_sentences[i] = []
        if type(sentence) == str:
            for word in word_tokenize(sentence):
                if word not in stop_words and type(word) == str:
                    words.update([word.lower()])  # Converting all the words to lowercase
                    tmp = re.sub('[^a-zA-Z]', "", word.lower())
                    train_sentences[i].append(tmp)
                    help_find_word[tmp] = False
            if (i + 1) % 24000 == 0:
                print('tokenize %d%% done in training dataset' % ((i + 1) * 100 / len(sentences)))

    print('Loading pre-trained word-embedding model...')
    embeddings_dict = {}
    with open("./data/glove.6B.300d.txt", 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            help_find_word[word] = True
            vector = np.asarray(values[1:], "float32")
            vector = torch.FloatTensor(vector)
            embeddings_dict[word] = vector

    print('Padding training dataset')
    train_sentences = pad_input(train_sentences, help_find_word)
    print('Word embedding in training dataset by pre-trained model')
    train_data = word_embed(train_sentences, embeddings_dict)
    return train_data, train_labels, help_find_word, embeddings_dict


def preprocessing_test_data(test, dataset_name, help_find_word, embeddings_dict):
    """
    test: the test dataset, and the data type is pandas dataframe.
    dataset_name: the dataset name, and the data type is string.
    help_find_word: map known word to true, map unknown word to false.
    embedding_dict: map a word to a 300d vector.
    return:
        test_data: a list of sentences have been processed.
        test_labels: an array that contains the true labels of testing dataset.
    """
    test[0] = test[0] - 1
    test_labels = test[0].to_numpy()
    test['pad'] = ' '
    if dataset_name != 'yelp_review_polarity_csv':
        test['sentences'] = test[1] + test['pad'] + test[2]
    else:
        test['sentences'] = test[1]
    test_sentences = [None] * len(test['sentences'])
    for i, sentence in enumerate(test['sentences']):
        test_sentences[i] = []
        if type(sentence) == str:
            for word in word_tokenize(sentence):
                test_sentences[i].append(word.lower())
            if (i + 1) % 1520 == 0:
                print('tokenize %d%% done in testing dataset' % ((i + 1) * 100 / len(test['sentences'])))
    print('Padding testing dataset')
    test_sentences = pad_input(test_sentences, help_find_word)
    print('Word embedding in testing dataset by pre-trained model')
    test_data = word_embed(test_sentences, embeddings_dict)
    return test_data, test_labels

def generateTrainData(dataset_name, data_size):
    path = './data/' + dataset_name + '/train.csv'
    train = pd.read_csv(path, header=None)
    random_ind = np.random.permutation(len(train))
    train = train.iloc[random_ind][:int(data_size * 120000)]
    train_data, train_labels, help_find_word, embedding_dicts = preprocessing_train_data(train, dataset_name)
    return train_data, train_labels, help_find_word, embedding_dicts

def generateTestData(dataset_name, help_find_word, embedding_dicts):
    path = './data/' + dataset_name + '/test.csv'
    test = pd.read_csv(path, header=None)
    random_ind = np.random.permutation(len(test))
    test = test.iloc[random_ind][:7600]
    test_data, test_labels = preprocessing_test_data(test, dataset_name, help_find_word, embedding_dicts)
    return test_data, test_labels

def load_dataset(dataset_name=None, batch_size=400, data_size=1.0):
    """
    dataset_name: the dataset that user want to use. Default is AGNews
    return:
        vocab_size: the number of vocabulary in the dataset.
        word_embeddings: word embedding that we used, which is a vector.
        train_loader: DataLoader for training set.
        val_loader: DataLoader for validation set.
        test_loader: DataLoader for testing set.
    """

    if dataset_name is None or dataset_name == 'AGNews':
        dataset_name = 'ag_news_csv'
    elif dataset_name == 'Yelp':
        dataset_name = 'yelp_review_polarity_csv'
    elif dataset_name == 'Amazon':
        dataset_name = 'amazon_review_polarity_csv'
    elif dataset_name == 'Dbpedia':
        dataset_name = 'dbpedia_csv'

    train_data, train_labels, help_find_word, embedding_dicts = generateTrainData(dataset_name, data_size)
    test_data, test_labels = generateTestData(dataset_name, help_find_word, embedding_dicts)

    vocab_size = len(train_data)
    train_size = int(0.8 * len(train_data))
    training_dataset = TensorDataset(train_data, torch.from_numpy(train_labels))
    train_dataset, val_dataset = torch.utils.data.random_split(training_dataset, [train_size, len(training_dataset) - train_size])
    test_dataset = TensorDataset(test_data, torch.from_numpy(test_labels))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print("Finished data loading.")
    return vocab_size, train_loader, val_loader, test_loader

if __name__ == '__main__':
    load_dataset('AGNews', data_size=0.2)
    pass