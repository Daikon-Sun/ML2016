import csv
import numpy as np
import matplotlib.pyplot as plt

def train_minibatch(dat, a, c, w, b, ada_da, ada_ga, ada_dc, ada_gc, ada_dw, ada_gw, ada_db, ada_gb, rho, epsilon, lmbd):
    '''
    renew parameters one time with regard to dat,
    which is one minibatch of data
    '''
    res = [-b[0] for j in range(480-mxcnt)]
    for i in range(mxcnt, 480, 1):
        res[i-mxcnt] += dat[pm25, i]
        for j in range(len(a)):
            res[i-mxcnt] -= a[j]*dat[pm25, i-j-1]*dat[pm25, i-j-1]
        for j in range(len(c)):
            res[i-mxcnt] -= c[j]*dat[pm10, i-j-1]*dat[pm10, i-j-1]
        for j in range(len(name)):
            for k in range(para_cnt[j]):
                res[i-mxcnt] -= w[j][k]*dat[ pos[j], i-k-1]

    da = [ 0 for i in range(len(a)) ]
    gra_a = [ 0 for i in range(len(a)) ]
    for i in range(len(a)):
        for j in range(mxcnt, 480, 1):
            gra_a[i] -= 2*res[j-mxcnt]*dat[pm25, j-i-1]*dat[pm25, j-i-1]
        gra_a[i] += 2*lmbd*a[i]
        ada_ga[i] = ada_ga[i]*rho+(1.-rho)*(gra_a[i]**2)
        da[i] = -gra_a[i]*(ada_da[i]+epsilon)**0.5/(ada_ga[i]+epsilon)**0.5
        ada_da[i] = rho*ada_da[i]+(1.-rho)*(da[i]**2)
        a[i] += da[i]

    dc = [ 0 for i in range(len(c)) ]
    gra_c = [ 0 for i in range(len(c)) ]
    for i in range(len(c)):
        for j in range(mxcnt, 480, 1):
            gra_c[i] -= 2*res[j-mxcnt]*dat[pm10, j-i-1]*dat[pm10, j-i-1]
        gra_c[i] -= 2*lmbd*c[i]
        ada_gc[i] = ada_gc[i]*rho+(1.-rho)*(gra_c[i]**2)
        dc[i] = -gra_c[i]*(ada_dc[i]+epsilon)**0.5/(ada_gc[i]+epsilon)**0.5
        ada_dc[i] = rho*ada_dc[i]+(1.-rho)*(dc[i]**2)
        c[i] += dc[i]

    dw = []
    gra_w = []
    for i in range(len(name)):
        gra_w.append(list([] for j in range(para_cnt[i])))
        dw.append(list([] for j in range(para_cnt[i])))
        for j in range(para_cnt[i]):
            tgra = 0.0
            for k in range(mxcnt, 480, 1):
                tgra -= 2*res[k-mxcnt]*dat[ pos[i], k-j-1]
            gra_w[i][j] = tgra+2*lmbd*w[i][j]
            ada_gw[i][j] = ada_gw[i][j]*rho+(1.-rho)*(gra_w[i][j]**2)
            dw[i][j] = -gra_w[i][j]*(ada_dw[i][j]+epsilon)**0.5/(ada_gw[i][j]+epsilon)**0.5
            ada_dw[i][j] = rho*ada_dw[i][j]+(1.-rho)*(dw[i][j]**2)
            w[i][j] += dw[i][j]

    gra_b = -2*sum(res)
    ada_gb[0] = ada_gb[0]*rho+(1.-rho)*(gra_b**2)
    db = -gra_b*(ada_db[0]+epsilon)**0.5/(ada_gb[0]+epsilon)**0.5;
    ada_db[0] = rho*ada_db[0]+(1.-rho)*(db**2)
    b[0] += db
def one_predict(dat, a, c, w, b):
    prd = b[0]
    for i in range(len(name)):
        for j in range(para_cnt[i]):
            prd += w[i][j]*dat[ pos[i], -j-1]
    for i in range(len(a)):
        prd += a[i]*dat[pm25, -i-1]*dat[pm25, -i-1]
    for i in range(len(c)):
        prd += c[i]*dat[pm10, -i-1]*dat[pm10, -i-1]
    return prd

def do_predict(a, c, w, b):
    '''
    predict the outcome of test_X.csv
    '''
    predict = []
    for t in range(240):
        predict.append(one_predict(arrx[:,9*t:9*(t+1)], a, c, w, b))
    return predict

#add new category of data
def add_data(nm, cnt):
    name.append(nm)
    para_cnt.append(cnt)
    pos.append([])
    global mxcnt
    mxcnt = max(mxcnt, cnt)

def training(itnum):
    '''
    complete training with itnum iterations
    '''
    global All_data
    a = [0, 0, 0, 0]
    c = [0]
    w = []
    b = [0]
    ada_da = [0, 0, 0, 0]
    ada_ga = [0, 0, 0, 0]
    ada_dc = [0]
    ada_gc = [0]
    ada_dw = []
    ada_gw = []
    ada_db = [0]
    ada_gb = [0]
    for i in range(len(name)):
        w.append([0.0 for j in range(para_cnt[i])])
        ada_dw.append([0.0 for j in range(para_cnt[i])])
        ada_gw.append([0.0 for j in range(para_cnt[i])])
    mon_seq = np.arange(12)
    for t in range(1, itnum+1, 1):
        #np.random.shuffle(mon_seq)
        for i in range(12):
            train_minibatch(
                All_data[:, mon_seq[i]*480:(mon_seq[i]+1)*480],
                a, c, w, b,
                ada_da, ada_ga, ada_dc, ada_gc,
                ada_dw, ada_gw, ada_db, ada_gb,
                0.95, 1e-8, 0)
        print(t)
        if t == itnum:
            predict = do_predict(a, c, w, b)
            with open('linear_regression.csv', 'w') as csvfile:
                wrtr = csv.writer(csvfile)
                wrtr.writerow(['id', 'value'])
                for i in range(240):
                    wrtr.writerow(['id_'+str(i)]+[predict[i]])
            #for i in range(len(a)):
            #    print('a', i, a[i])
            #for i in range(len(c)):
            #    print('c', i, c[i])
            #for i in range(len(name)):
            #    print(name[i])
            #    for j in range(para_cnt[i]):
            #        print(j, w[i][j])
            #print('b', b[0])

All_data = []
ddict = {}
name = []
pos = []
para_cnt = []
arrx = [[] for i in range(18)]
rmse_his = []

#data initialization
myid = 0
f = open("train.csv", "r", encoding = 'big5')
csv_f = csv.reader(f)
next(csv_f)
for row in csv_f:
    if row[2] not in ddict:
        ddict[ row[2] ] = myid;
        myid += 1
        All_data.append([])
    if row[2] == 'RAINFALL':
        for i in range(3, len(row)):
            if(row[i] == 'NR'):
                All_data[ ddict[ row[2] ] ].append(0.0)
            else:
                All_data[ ddict[ row[2] ] ].append(float(row[i]))
    else:
        for i in range(3, len(row)):
            All_data[ ddict[ row[2] ] ].append(float(row[i]))
f.close()

All_data = np.array(All_data)   #from list to numpy array
pm25 = ddict[ 'PM2.5' ]
pm10 = ddict[ 'PM10' ]
o3 = ddict[ 'O3' ]
no2 = ddict[ 'NO2' ]

#parse test_X.csv to arrx
with open('test_X.csv', 'r', encoding = 'big5') as csvfile:
    csv_f = csv.reader(csvfile)
    for row in csv_f:
        if row[1] == 'RAINFALL':
            for i in range(2, len(row)):
                if row[i] == 'NR':
                    arrx[ ddict[row[1]] ].append(0.0)
                else:
                    arrx[ ddict[row[1]] ].append(float(row[i]))
        else:
            for i in range(2, len(row)):
                arrx[ ddict[row[1]] ].append(float(row[i]))

arrx = np.array(arrx)

mxcnt = 0

add_data('PM2.5', 9)
add_data('PM10', 5)
add_data('O3', 3)
add_data('RAINFALL', 2)
add_data('NO2', 1)
for i in range(len(name)):  #get pos of every category of data
    pos[i] = ddict[ name[i] ]
accum = sum(para_cnt)

training(1100)
