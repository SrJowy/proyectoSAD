import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from nltk.tokenize import RegexpTokenizer
import numpy as np


data = pd.read_csv("nuevos_datos.csv", sep=',', dtype={"col_0": "string", "asin": "string", "brand": "string", "price": "string", "title":"string", "col_5":"string", "category": "string", "rank": "string", "description": "string", "main_cat": "string", "details": "string", "overall": int, "vote_y": "string", "verified_y": bool, "reviewTime_y": "string", "reviewerID_y": "string", "reviewerName_y": "string", "reviewText": "string", "summary": "string", "unixReviewTime": int, "State": "string", "Country": "string", "Postal Code": int})
#print(data.head())
#print(data.info())

token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True, stop_words="english", ngram_range= (1,1), tokenizer = token.tokenize)
X = cv.fit_transform(data['reviewText'].values.astype('U')).toarray()

y = data["overall"] #Select the stars column

X_train, X_test, y_train, y_test = train_test_split(X, y)

classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(f1_score(y_test, y_pred, average='macro'))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))