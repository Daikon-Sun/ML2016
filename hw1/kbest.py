import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from math import sqrt

def train_minibatch(dat, w, b, lr, lmbd, ada_w, ada_b, with_ada, with_reg):
    '''
    renew parameters one time with regard to dat,
    which is one minibatch of data
    '''
    res = [-b[0] for j in range(480-mxcnt)]
    for i in range(mxcnt, 480, 1):
        res[i-mxcnt] += dat[pm25, i]
        for j in range(len(name)):
            for k in range(para_cnt[j]):
                res[i-mxcnt] -= w[j][k]*dat[ pos[j], i-k-1]
    gra_w = []
    gra_b = -2*sum(res)
    for i in range(len(name)):
        gra_w.append([])
        for j in range(para_cnt[i]):
            tgra = 0.0
            for k in range(mxcnt, 480, 1):
                tgra -= 2*res[k-mxcnt]*dat[ pos[i], k-j-1]
            gra_w[i].append(tgra+2*lmbd*w[i][j])
    for i in range(len(name)):
        for j in range(para_cnt[i]):
            ada_w[i][j] += gra_w[i][j]**2
            w[i][j] -= lr*gra_w[i][j]/sqrt(ada_w[i][j])
    ada_b[0] += gra_b**2
    b[0] -= lr*gra_b/sqrt(ada_b[0])

def one_predict(dat, w, b):
    prd = b[0]
    for i in range(len(name)):
        for j in range(para_cnt[i]):
            prd += w[i][j]*dat[ pos[i], -j-1]
    return prd

def do_predict(w, b):
    '''
    predict the outcome of test_X.csv
    '''
    predict = []
    for t in range(240):
        predict.append(one_predict(arrx[:,9*t:9*(t+1)], w, b))
    return predict

#add new category of data
def add_data(nm, cnt):
    name.append(nm)
    para_cnt.append(cnt)
    pos.append([])
    global mxcnt
    mxcnt = max(mxcnt, cnt)

def training(itnum, lr):
    '''
    complete training with itnum iterations
    '''
    w = []
    ada_w = []
    b = [0]
    ada_b = [0]
    for i in range(len(name)):
        w.append([0.0 for j in range(para_cnt[i])])
        ada_w.append([0.0 for j in range(para_cnt[i])])
    for t in range(1, itnum+1, 1):
        for i in range(12):
            train_minibatch(All_data[:, i*480:(i+1)*480], w, b, lr, 0, ada_w, ada_b, 0, 0)
        if t == itnum:
            predict = do_predict(w, b)
            with open('kaggle_best.csv', 'w') as csvfile:
                wrtr = csv.writer(csvfile)
                wrtr.writerow(['id', 'value'])
                for i in range(240):
                    wrtr.writerow(['id_'+str(i)]+[predict[i]])
            #for i in range(len(name)):
            #    print(name[i])
            #    for j in range(para_cnt[i]):
            #        print(j, w[i][j])
            #print('b ', b[0])

All_data = []
ddict = {}
name = []
pos = []
para_cnt = []
pm25 = 0
arrx = [[] for i in range(17)]

#data initialization
myid = 0
f = open("train.csv", "r", encoding = 'big5')
csv_f = csv.reader(f)
next(csv_f)
for row in csv_f:
    if row[2] == 'RAINFALL':
        continue
    if row[2] not in ddict:
        ddict[ row[2] ] = myid;
        myid += 1
        All_data.append([])
    for i in range(3, len(row)):
        All_data[ ddict[ row[2] ] ].append(float(row[i]))
f.close()

#parse test_X.csv to arrx
with open('test_X.csv', 'r', encoding = 'big5') as csvfile:
    csv_f = csv.reader(csvfile)
    for row in csv_f:
        if row[1] == 'RAINFALL':
            continue
        for i in range(2, len(row)):
            arrx[ ddict[row[1]] ].append(float(row[i]))

pm25 = ddict[ 'PM2.5' ]

mxcnt = 0

add_data('PM2.5', 9)
#add_data('PM10', 3) #0.776426
#add_data('O3', 2)   #0.35667
#add_data('CH4', 2)  #0.254657
#add_data('CO', 2)   #0.283119
#add_data('NMHC', 2) #0.291778
#add_data('AMB_TEMP', 1) #-0.017127
#add_data('NO', 2)   #0.02997
#add_data('NO2', 2)  #0.449113
#add_data('NOx', 1)  #0.375564
#add_data('RH', 1)   #-0.264196
#add_data('SO2', 2)  #0.370831
#add_data('THC', 2)  #0.352159
#add_data('WD_HR', 1)    #0.186138
#add_data('WIND_DIREC', 1)   #0.15699
#add_data('WIND_SPEED', 1)   #-0.084703
#add_data('WS_HR', 1)    #-0.045458

for i in range(len(name)):  #get pos of every category of data
    pos[i] = ddict[ name[i] ]
accum = sum(para_cnt)

All_data = np.array(All_data)   #from list to numpy array
arrx = np.array(arrx)

training(10000, 0.1)
