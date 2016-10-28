import csv
import numpy as np
import sys

f = open(sys.argv[1], 'rb')
data_mean = np.load(f, None, False)
data_std = np.load(f, None, False)
idx_list = np.load(f, None, False).tolist()
shapes = np.load(f, None, False)
fn_idx = np.load(f, None, False)

all_test_data = []
with open(sys.argv[2], 'r') as csvfile:
    csvf = csv.reader(csvfile)
    for row in csvf:
        tmp = []
        for i in range(1, len(row)):
            if i-1 in idx_list:
                tmp.append([(float(row[i])-data_mean[i-1, 0])/data_std[i-1, 0]])
        all_test_data.append(np.array(tmp))

class sigmoid:
    @staticmethod
    def f(x):
        return 1.0/(1.0+np.exp(-x))
class tanh:
    @staticmethod
    def f(x):
        return np.tanh(x)
class arctan:
    @staticmethod
    def f(x):
        return np.arctan(x)
class relu:
    @staticmethod
    def f(x):
        return np.maximum(0, x)
class leaky_relu:
    @staticmethod
    def f(x):
        return alpha*x*(x<=0)+np.maximum(0, x)

W = []
B = []
for i in range(len(shapes)-1):
    tmp_w = np.load(f, None, False)
    W.append(tmp_w)
    tmp_b = np.load(f, None, False)
    B.append(tmp_b)
f.close()

if fn_idx[0, 0] == 0:
    fn = sigmoid()
elif fn_idx[0, 0] == 1:
    fn = tanh()
elif fn_idx[0, 0] == 2:
    fn = atctan()
elif fn_idx[0, 0] == 3:
    fn = relu()
elif fn_idx[0, 0] == 4:
    alpha = fn_idx[0, 1]
    fn = leaky_relu()

outcome = []
for tst in all_test_data:
    a = tst
    for w, b in zip(W, B):
        a = fn.f(np.dot(w, a)+b)
    outcome.append(np.argmax(a))

with open(sys.argv[3], 'w') as csvfile:
    wrtr = csv.writer(csvfile)
    wrtr.writerow(['id']+['label'])
    for i in range(len(outcome)):
        if outcome[i] == 1:
            wrtr.writerow([i+1]+[1])
        else:
            wrtr.writerow([i+1]+[0])
