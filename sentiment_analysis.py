# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 10:55:03 2019

@author: HarshGB
"""
import numpy as np
import pandas as pd
from collections import Counter
import nltk.tokenize as token
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.feature_extraction import DictVectorizer




amazon = pd.read_csv(r"C:\Users\harshgb\Desktop\amazon_baby.csv", na_values = ' ')
#print(amazon)
amazon = pd.DataFrame(amazon)
#print(amazon)
print('\n')
'''
review = amazon['review']
print(review)
'''
'''
count = [Counter(review)]
print(count)
'''
amazon['review'] = amazon['review'].fillna('NA')
amazon['word_count'] = amazon['review'].apply(lambda x: Counter(x.split(' ')))
#print(amazon['word_count'])
#print(amazon.head())
#plt.hist(amazon['name'])

amazon = amazon[amazon['rating'] != 3]
#print(amazon)

amazon['sorted_rating'] = amazon['rating'].apply(lambda x: 1 if x >= 4 else 0)

amazon['word_count'] = DictVectorizer(sparse = False).fit_transform(amazon['word_count'])



X = amazon['word_count']
y = amazon['sorted_rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1,
                                                    test_size = 0.80)
'''
LogisticRegression.fit(X_train,y_train)
rfe_pred = LogisticRegression.predict(X_test)
'''
logit = LogisticRegression()
logit.fit(X_train, y_train)
