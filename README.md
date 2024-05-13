**Sentiment Analysis with NLTK**

This project aims to perform sentiment analysis on Twitter data using the NLTK (Natural Language Toolkit) library in Python.

**Overview**
Sentiment analysis is the process of determining the sentiment (positive, negative, or neutral) of a piece of text. In this project, we use the NLTK library to analyze tweets from the Sentiment140 dataset, which contains 1.6 million tweets labeled with sentiment polarity (0 for negative and 4 for positive).

**Requirements**
To run the code in this project, you will need:

Python 3.x
NLTK (Natural Language Toolkit)
Pandas
Numpy
You can install NLTK and other required packages using pip:

Copy code
pip install nltk pandas numpy
Steps
Data Preparation: We download the Sentiment140 dataset using the Kaggle API and extract a subset of 14,000 tweets for analysis.
Preprocessing: We preprocess the tweet text by converting it to lowercase, removing URLs, punctuation, user references, and stopwords. We also perform tokenization, stemming, and lemmatization to normalize the text.
Feature Extraction: We extract features from the preprocessed tweet text using the Bag-of-Words model. Each tweet is represented as a dictionary of word features, where the keys are the words present in the tweet and the values are True.
Model Training: We split the dataset into training and testing sets, and then train a Naive Bayes classifier using the NLTK library.
Model Evaluation: We evaluate the accuracy of the trained model on the testing dataset to assess its performance in classifying tweets as positive or negative.
Prediction: We demonstrate how to use the trained model to classify new tweets as either positive or negative.
Usage
To run the code:

Ensure you have installed Python and the required libraries (NLTK, Pandas, Numpy).
Download the Sentiment140 dataset from Kaggle (https://www.kaggle.com/kazanova/sentiment140) and extract the training.1600000.processed.noemoticon.csv file.
Run the provided Python script (sentiment_analysis.py) to preprocess the data, train the model, and perform sentiment analysis on custom tweets.
