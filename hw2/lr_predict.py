import csv
import numpy as np
import sys

all_test_data = []
with open(sys.argv[2], 'r') as csvfile:
    csvf = csv.reader(csvfile)
    for row in csvf:
        all_test_data.append([])
        for i in range(1, len(row)):
            all_test_data[-1].append(float(row[i]))
all_test_data = np.array(all_test_data)

w = []
idx_list = []
test_data = []
with open(sys.argv[1], 'r') as csvfile:
    csvf = csv.reader(csvfile)
    for row in csvf:
        w.append([float(row[1])])
        if int(row[0]) < 1000:
            test_data.append(all_test_data[:, int(row[0])])
        else:
            test_data.append([ 1. for i in range(len(all_test_data)) ])
test_data = np.transpose(np.array(test_data))
w = np.array(w)
outcome = np.dot(test_data, w)
with open(sys.argv[3], 'w') as csvfile:
    wrtr = csv.writer(csvfile)
    wrtr.writerow(['id']+['label'])
    for i in range(len(outcome)):
        if outcome[i] >= 0:
            wrtr.writerow([i+1]+[1])
        else:
            wrtr.writerow([i+1]+[0])
