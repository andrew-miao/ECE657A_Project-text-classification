# ECE657A Project: text classification
The project for the course ECE657A in spring 2020 at uwaterloo.

This repo contains several models, including CNN, RCNN, LSTM with attention, and Encoder-Decoder (LSTM as encoder, GRU as Decoder with attention mechansim), to solve the text-classification task.
We implement these models in Pytorch framework. 

The datasets that we used are AGNews, Dbpedia, Amazon reviews-Polarity, and Yelp reviews-Polarity. The last two
datasets are useful for sentiment analysis.

Here is the performance of these models in these datasets:

|Datasets      | CNN baseline | RCNN  | LSTM Attention | Encoder-Decoder |
| ------------- |:-------------:| :------------: | :------------: | :------------: 
| AGNews    | 78.48% | 90.85% | 90.23% | 88.67% |
| Dbpedia     | 84.73% |  96.57% | 95.88% | 96.26% |
| Amazon | 74.85% | 88.91% | 88.12% | 87.38% |
| Yelp | 75.29% | 85.12% | 85.20% | 85.45% |

## Requirements
> * Python == 3.7.6
> * Pytorch == 1.5.1
> * Numpy == 1.18.1

## Usage
Once you clone this repo, you can run the main.py to train the specify model for the dataset.

For example,

``python main.py lstm_attn agnews``

To simplify commands, we use 'cnn', 'rcnn', 'lstm_attn', 'lstm_gru' as the model names.
The datasets names are 'agnews', 'dbpeida', 'amazon', 'yelp'.