import csv

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


##FUNCTIONIZE##
#read in twitter data from csv, convert to data frame, split into train/test
infile = 'full-corpus.csv'
df = pd.read_csv(infile, header=0)


#create train and test sets
train_set, test_set = train_test_split(df, test_size = 0.3)
print(train_set.columns.values)

#create count vector
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = 'english',   \
                             max_features = 5000)

counts = vectorizer.fit_transform(train_set["TweetText"])

#create and train classifier
classifier = MultinomialNB()
targets = train_set["Sentiment"]

classifier.fit(counts, targets)

example_counts = vectorizer.transform(['I loved this new product, it was fantastic!', 'I hate Microsoft, they suck!!', 'bloop'])
predictions = classifier.predict(example_counts)
print(predictions) # [1, 0]
#training_words = df["TweetText"].tolist()

#print(training_words[:5])





