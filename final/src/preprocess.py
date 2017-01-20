import pandas as pd
import sys
import numpy as np
from collections import Counter

train = pd.read_csv('data/train', header=None)
test = pd.read_csv('data/test.in', header=None)
print(train.head(10))
types = open('data/training_attack_types').read().splitlines()
types = dict([ t.split() for t in types ]+[('normal', 'normal')])
ddict = dict([ ('normal', 0), ('dos', 1), ('u2r', 2), ('r2l', 3), ('probe', 4) ])

label = train[41]
label = np.array([ [ddict[ types[ lbl[:-1] ] ]] for lbl in label ])
print(label.shape)

train.drop([41], axis=1, inplace=True)

trn_len = len(train[0])
tst_len = len(test[0])
print(trn_len)
print(tst_len)

alldata = pd.concat([train, test])
del train, test

def to_one_hot(idx):
    global alldata
    a = pd.get_dummies(alldata[idx], prefix=str(idx))
    alldata.drop([idx], axis=1, inplace=True)
    alldata = pd.concat([alldata, a], axis=1, ignore_index=True)
    del a

discrete_col = [3, 2, 1]
for col in discrete_col:
    to_one_hot(col)
    print(col)

a = alldata.columns
print(a)
print(len(a))
data = alldata.as_matrix()
print(data.shape)
np.save('expand_test', data[trn_len:])
print(data[:trn_len, :].shape)
np.save('expand_train', np.hstack((data[:trn_len, :], label)))
del alldata
