#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 15:34:55 2018

@author: elamrily
"""

import numpy as np
import os
from sklearn.model_selection import train_test_split


classes = []
filenames = []
X = []
y = []

for i, classe in enumerate (os.listdir('C:/Users/Ilyas/Desktop/M2 SD/Text Analysis/nlp-labs/tobacco-lab/data/Tobacco3482-OCR')):
    inputFilepath = classe
    filename_w_ext = os.path.basename(inputFilepath)
    filename, file_extension = os.path.splitext(filename_w_ext)
    classes.append(filename)
    path = 'C:/Users/Ilyas/Desktop/M2 SD/Text Analysis/nlp-labs/tobacco-lab/data/Tobacco3482-OCR/' + filename
    
    for j, element in enumerate (os.listdir(path)):
        inputFilepath = element
        filename_w_ext = os.path.basename(inputFilepath)
        filename, file_extension = os.path.splitext(filename_w_ext)
        filenames.append(filename)
        
        path_file = path + '/' + element
                
        file = open(path_file,'r').read()
        
        X.append(file)
        y.append(i)

y = np.asarray(y)
X = np.asarray(X)

X_app, X_test, y_app, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_train_dev, y_train, y_train_dev = train_test_split(X_app, y_app, test_size=0.25)

from sklearn.feature_extraction.text import CountVectorizer
# Create document vectors
vectorizer = CountVectorizer(max_features=3000)
vectorizer.fit(X_train)
X_train_counts = vectorizer.transform(X_train)
X_val_counts = vectorizer.transform(X_train_dev)
X_test_counts = vectorizer.transform(X_test)

from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()
clf.fit(X_train_counts,y_train)
y_pred = clf.predict(X_val_counts)
score_val = clf.score(X_val_counts,y_train_dev)
print('score : ',score_val)

from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer().fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
X_val_tf = tf_transformer.transform(X_val_counts)
X_test_tf = tf_transformer.transform(X_test_counts)

clf = MultinomialNB()
clf.fit(X_train_tf,y_train)
y_pred = clf.predict(X_val_tf)
score_val = clf.score(X_val_tf,y_train_dev)
print('score : ',score_val)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
# YOUR CODE HERE
print(classification_report(y_test, y_pred))
print('matrice de confusion : \n',confusion_matrix(y_test, y_pred))
