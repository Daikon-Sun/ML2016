import numpy as np
import sys
import xgboost as xgb
from time import time

t0 = time()

test = np.load('expand_test.npy')
dtest = xgb.DMatrix( test )
del test

train = np.load('expand_train.npy')
labels = np.array(train[:, -1:])
trains = np.array(train[:, :-1])
del train

print('reading data spent ' + str(time()-t0) + ' seconds')
t0 = time()

pred = []
num_round = [35, 40, 35, 45, 35]
fac = [0.7, 4, 1.3, 1.1, 0.9]
mxdp = [14, 14, 14, 14, 14]
for i in range(5):
    tlbl = (labels == i)
    smp = float(np.sum(tlbl))
    smn = (len(tlbl)-smp)
    dtrn = xgb.DMatrix( trains, label=tlbl )
    del tlbl
    evll = [(dtrn, 'train'+str(i))]
    para = { 'max_depth':mxdp[i], 'silent':1, 'objective':'binary:logistic'\
            , 'eta':0.26, 'min_child_weight':5, 'scale_pos_weight':smn/smp\
            , 'subsample':0.8, 'tree_method':'exact'}
    bst = xgb.train(para, dtrn, num_round[i], evll)
    pred.append( bst.predict( dtest )*fac[i] )
    del dtrn, evll, bst
    print('done with ' + str(i) + ' within ' + str(time()-t0) + 'seconds')
    t0 = time()

print('predicting...')
pred = np.argmax(np.array(pred), axis=0)

import csv
with open('prediction.csv','w') as output:
    writer = csv.writer(output)
    writer.writerow(['id','label'])
    for i, m in enumerate(pred):
        writer.writerow([i+1, m])
