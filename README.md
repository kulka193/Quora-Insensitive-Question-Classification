This repository contains code to Implement a classification model to categorize insensitive questions from Quora. The dataset is provided by Quora for a [Kaggle Competition](https://www.kaggle.com/c/quora-insincere-questions-classification).

You can refer to the following [latest Kernel](https://www.kaggle.com/kulka193/kernel40498143be/output?scriptVersionId=9116957) that I ran.

## Dataset: 
Contains about 13M data samples from Quora's question base with over 250K unique words in the text corpus. The insensitive questions are labelled 1. 

## Model: 
The model consists of an Embedding layer to fetch the word embeddings for words each data sample. Then it has a combination of convolutional (Conv1D) layers and Max Pool (x3), LSTM units and two more dense(with dropout) layers. Finally a sigmoid layer is used to make a decision between the classes.

## Training:

The model was trained with ``batch size=256`` and the cross-entropy loss was optimized using a minibatch Adam optimizer. The training was done for about 10 epochs on a Tesla K80 to the Kaggle Kernel.  A 90/10 split was used on the training data further on the training data to evaluate on the validation data.

## Evaluation and Results:

The model was evaluated with a F-1 score=0.532 on the Validation data. 
