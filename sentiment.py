import csv

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

##Set corpus file here, as well as the headers of the text and labels
infile = 'Sentiment_Analysis_Dataset2.csv'
data = "SentimentText"
labels = "Sentiment"

##FUNCTIONIZE##
#read in twitter data from csv, convert to data frame, split into train/test
class Classifier():
    def __init__(self):
        self.train_set, self.test_set = self.load_data()
        self.counts, self.test_counts, self.vectorizer = self.vectorize()
        self.classifier = self.train_model()

    def load_data(self):
        df = pd.read_csv(infile, header=0, error_bad_lines=False)
        train_set, test_set = train_test_split(df, test_size=.3)
        return train_set, test_set

    def train_model(self):
        classifier = BernoulliNB()
        targets = self.train_set[labels]
        classifier.fit(self.counts, targets)
        return classifier

    #vectorizer takes the corpus text, tokenizes it, and counts
    #n-gram range (1,2) counts unigrams and bigrams
    def vectorize(self):
        vectorizer = TfidfVectorizer(min_df=5,
                                 max_df = 0.8,
                                 sublinear_tf=True,
                                 ngram_range = (1,2),
                                 use_idf=True)
        counts = vectorizer.fit_transform(self.train_set[data])
        test_counts = vectorizer.transform(self.test_set[data])

        return counts, test_counts, vectorizer


    def evaluate(self):
        test_counts,test_set = self.test_counts, self.test_set
        predictions = self.classifier.predict(test_counts)
        print (classification_report(test_set[labels], predictions))
        print ("The accuracy score is {:.2%}".format(accuracy_score(test_set[labels], predictions)))

    def classify(self, input):
        input_text = input
        input_counts = self.vectorizer.transform(input_text)
        predictions = self.classifier.predict(input_counts)
        print(predictions)

myModel = Classifier()

text = ['good great happy like', 'bad awful this sucks']

myModel.classify(text)
myModel.evaluate()
