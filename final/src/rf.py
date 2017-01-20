import numpy as np
import sys, csv
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

tree_num = 27

test = np.load('expand_test.npy')
data = np.load('deduplicate_train.npy')
train = data[:, :-1]
label = data[:, -1]

clf = RandomForestClassifier( n_estimators=tree_num, n_jobs=-1 )
clf.fit( train, label )
pred = clf.predict(test)

with open('basic.csv','w') as output:
    writer = csv.writer(output)
    writer.writerow(['id','label'])
    for i, m in enumerate(pred):
        writer.writerow([i+1, int(m)])
