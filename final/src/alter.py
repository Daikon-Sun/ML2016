import numpy as np
import sys, csv
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

tree_num = 25

test = np.load('test.npy')
data = np.load('deduplicate_train.npy')
train = data[:, :-1]
label = data[:, -1]

pred = pd.read_csv('prediction.csv').as_matrix()[:, 1]

tpred = np.copy(pred)
clf = RandomForestClassifier( n_estimators=tree_num, n_jobs=-1 )
clf.fit( train, label )
prob = clf.predict_proba(test)

for j in range(len(tpred)):
    if prob[j][3] > 0:
        tpred[j] = 3

with open('best.csv','w') as output:
    writer = csv.writer(output)
    writer.writerow(['id','label'])
    for i, m in enumerate(tpred):
        writer.writerow([i+1, int(m)])
