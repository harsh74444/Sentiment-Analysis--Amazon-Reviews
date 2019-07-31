# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 10:55:03 2019

@author: HarshGB
"""
import pandas as pd
from collections import Counter
import sklearn as skl
import nltk.tokenize as token

amazon = pd.read_csv(r"C:\Users\harshgb\Desktop\amazon_baby.csv", na_values = ' ')
print(amazon)
amazon = pd.DataFrame(amazon)
print(amazon)
print('\n')
review = amazon['review']
print(review)
'''
count = [Counter(review)]
print(count)
'''
amazon['review'] = amazon['review'].fillna('NA')
amazon['word_count'] = amazon['review'].apply(lambda x: Counter(x.split(' ')))
#print(amazon['word_count'])
#print(amazon.head())

