# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 16:08:12 2020

@author: giho9
"""

import pandas as pd
import numpy as np
import os

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

def setup(df, path, value):
    folder = os.listdir(path)

    economy = [pd.read_csv(path + '/' + file) for file in folder]

    for index in economy: 
            df = pd.merge(df, index[index.columns[0::2]], on='Date')

    date = df['Date']

    origin = df[df.columns[:9]]
    index = df[df.columns[9:]].shift(value).dropna(axis=0)

    index['Date'] = date

    dataset = pd.merge(origin, index, on='Date')

    features = df.columns[9:]

    return dataset, features

def score(test_y, pred_y):
    tp, fp, fn, tn = 0, 0, 0, 0
    for x,y in zip(test_y, pred_y):
        if y == 1 and x == 1: tp += 1
        if y == 1 and x == 0: fp += 1
        if y == 0 and x == 1: fn += 1
        if y == 0 and x == 0: tn += 1

    try:
        acc = round((tp + tn) / len(pred_y), 3)
        pre = round(tp / (tp + fp), 3)
        rec = round(tp / (tp + fn), 3)   
    except(ZeroDivisionError):
        return 0, 0, 0
    return acc, pre, rec

if __name__ == "__main__":
    folder = 'independent'
    
    target = ['LM4DN']
    
    df = pd.read_csv('dependent/^IXIC.csv')

    data, features = setup(df, folder, 3)
    
    train_size = int(len(data) * 0.69)
    
    clf = KNeighborsClassifier()
    
    param_grid = {
                'n_neighbors' : [2,3,4,5,6,7,8,9],
                'weights' : ['uniform'],
                }
    
    feature = ['CSUSHPINSArate', 'NEWORDERrate', 'PPIACOrate']
    
    X, y = \
        np.array(pd.DataFrame(data, columns=feature)), \
        np.array(pd.DataFrame(data, columns=target))
    
    train_x, train_y, test_x, test_y = \
        X[:train_size], y[:train_size], X[train_size:], y[train_size:]
        
    gcv=GridSearchCV(clf, param_grid=param_grid, scoring='f1')
    
    gcv.fit(train_x, train_y.ravel())
    
    pred_y = gcv.predict(test_x)
    
    report = score(test_y, pred_y)
  
    print('accuracy :',report[0])
    print('precision :',report[1])
    print('recall :',report[2])
    
    data = data[-len(pred_y):]

    data['predict'] = pred_y
    
    data.to_csv('predict.csv', index=False)
    
    