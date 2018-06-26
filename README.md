# Kaggle Research Paper Classification Challenge

## Overview

This repository contains the code for my models for a private machine learning Kaggle competition. The competition objective was to create a multilabel classifier that could classify the provided papers on the journal they were published on based on the title, abstract and a graph of citations among the papers.

The competition submissions were evaluated based on the log loss of the predicted vs the actual classes.

My submissions were part of the two person winning team, for which we used a fine tuned ensemble model.

Two different models were created one based on an LSTM neural network that was fed the word embeddings of the combined title and anstract of the paper and a few other features, and a second model based on Convolutional layers with three different filter sizes (3, 7, 11) and again a few more features.

## Installing Requirements

The code requirements are contained in the requirements txt files and can be installed via the pip tool with `pip install -r requirements.txt` or `pip install -r requirements-gpu.txt` for the gpu equiped machines. Please notice you need to also have installed CUDA and cuDNN, and the current version of Keras and Tensorflow are working with CUDA version 8.0 and cuDNN 6.0.

Furthermore the [`DeepWalk`](https://github.com/phanein/deepwalk) library was used to generate the deep-walk features from the Graph, so be also have it installed.

### Further Requirements

Both models used the pre-trained word embeddings from the Stanform [GloVe](https://nlp.stanford.edu/projects/glove/) project. These are required to run the model and can be downloaded [here](http://nlp.stanford.edu/data/glove.42B.300d.zip).

## Models

### LSTM-based

The first model that was build and the one that achieved the best performance on its own, had the following architecture:
![LSTM](models/LSTMClassifier/model.png)

We can see that the combined text of the paper title and abstract is passed through the Embedding layer (where the training feature for the words is set to True). This produces the word embeddings which are then passed through a Dropout layer (because training the embeddings tends to produce overfitting), and finally fed into an LSTM. 

After trying different architectures for the LSTM the one that produced the best result in this problem was for the LSTM to return the result of the word embedding processing at each step and them use a Max Pooling layer to select the max of those as the output.

The second feature that was used is an Embedding representation of the paper authors. All the authors in the papers were transformed into a unique integer and these are fed into an Embedding layer. The layer during training produces vectors that close in space when they are writing on the same journals.

The third feature is a 3-dimensional vector that has the normalized number of incoming/outgoing degrees to each paper, and the average neighbor degree.

The fourth and last feature is a 64-dimensional vector generated as a random walk starting from the current node and continuing to a random neighbor of the node each time. This is based on the [DeepWalk](https://arxiv.org/abs/1403.6652) paper.


### CNN-based

The second model that was developed used the following architecture based on 1-dimensional Convolutional layers:
![CNN](models/CNNClassifier/model.png)

As can be seen there were three different convolutional layers that were fed the word embeddings, with the following filters: 3, 7, 11. This created different feature maps that (in theory) represented different abstractions of the input text. These were then fed into Average Pooling layers that compute the means of these feature maps and produce the relevant outputs. The three outputs are then concatenated into a single 3x larger feature.

The second feature in this model was based on embedding representations of each paper's adjacent papers. Each paper is transformed into a unique integer and then passed through the Embedding layer to produce the final feature.

The third and fourth features used are the same as in the LSTM model.

## Data Loader

Finally a distinct module (`data_loader.py`) was created to generate the data numpy arrays that are fed for training, validation and testing. This was deemed necessary because of the time it took to load the data, especially the pre-trained GloVe embeddings which are about 2.5GB in size.
