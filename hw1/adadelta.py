import csv
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

class LR_model:

    def __init__(self, train_data, array_x, dictionary, mylmbd, myrho, myepsilon):
        self.trn_dat = train_data
        self.arrx = array_x
        self.ddict = dictionary
        self.lmbd = mylmbd
        self.rho = myrho
        self.epsilon = myepsilon
        self.name = []
        self.para_cnt = []
        self.pos = []
        self.mxcnt = 0
        self.b = [0.0]
        self.ada_b = [0.0]
        self.pm25 = ddict[ 'PM2.5' ]
        self.pm10 = ddict[ 'PM10' ]
        self.w = []
        self.ada_dw = []
        self.ada_gw = []
        self.ada_db = [0]
        self.ada_gb = [0]

    def add_data(self, nm, cnt):
        self.name.append(nm)
        self.para_cnt.append(cnt)
        self.pos.append( self.ddict[ nm ] )
        self.mxcnt = max(self.mxcnt, cnt)
        self.w.append([0.0 for i in range(cnt)])
        self.ada_dw.append([0.0 for i in range(cnt)])
        self.ada_gw.append([0.0 for i in range(cnt)])

    def train_minibatch(self, dat):
        '''
        renew parameters one time with regard to dat,
        which is one minibatch of data
        '''
        res = [-self.b[0] for j in range(480-self.mxcnt)]
        for i in range(self.mxcnt, 480, 1):
            res[i-self.mxcnt] += dat[self.pm25, i]
            for j in range(len(self.name)):
                for k in range(self.para_cnt[j]):
                    res[i-self.mxcnt] -= self.w[j][k]*dat[ self.pos[j], i-k-1]

        dw = []
        gra_w = []
        gra_b = -2*sum(res)
        for i in range(len(self.name)):
            gra_w.append(list([] for j in range(self.para_cnt[i])))
            dw.append(list([] for j in range(self.para_cnt[i])))
            for j in range(self.para_cnt[i]):
                tgra = 0.0
                for k in range(self.mxcnt, 480, 1):
                    tgra -= 2*res[k-self.mxcnt]*dat[ self.pos[i], k-j-1]
                gra_w[i][j] = (tgra+2*self.lmbd*self.w[i][j])
                self.ada_gw[i][j] = self.ada_gw[i][j]*self.rho+(1.-self.rho)*(gra_w[i][j]**2)
                dw[i][j] = -gra_w[i][j]*sqrt(self.ada_dw[i][j]+self.epsilon)/sqrt(self.ada_gw[i][j]+self.epsilon)
                self.ada_dw[i][j] = self.rho*self.ada_dw[i][j]+(1.-self.rho)*(dw[i][j]**2)
                self.w[i][j] += dw[i][j]
        self.ada_gb[0] = self.ada_gb[0]*self.rho+(1.-self.rho)*(gra_b**2)
        db = -gra_b*sqrt(self.ada_db[0]+self.epsilon)/sqrt(self.ada_gb[0]+self.epsilon);
        self.ada_db[0] = self.rho*self.ada_db[0]+(1.-self.rho)*(db**2)
        self.b[0] += db

    def one_predict(self, dat):
        prd = self.b[0]
        for i in range(len(self.name)):
            for j in range(self.para_cnt[i]):
                prd += self.w[i][j]*dat[ self.pos[i], -j-1]
        return prd

    def output_predict(self):
        predict = []
        for i in range(240):
            predict.append(self.one_predict(self.arrx[:,9*i:9*(i+1)]))
        with open('linear_regression.csv', 'w') as csvfile:
            wrtr = csv.writer(csvfile)
            wrtr.writerow(['id', 'value'])
            for i in range(240):
                wrtr.writerow(['id_'+str(i)]+[predict[i]])

    def training(self, itnum):
        '''
        complete training with itnum iterations
        '''
        for t in range(1, itnum+1, 1):
            self.train_minibatch(self.trn_dat)

    def output_parameters(self):
        for i in range(len(self.name)):
            print(self.name[i])
            for j in range(self.para_cnt[i]):
                print(j, self.w[i][j])
        print('b ', self.b[0])

ddict = {}
train_data = []
myid = 0
f = open("train.csv", "r", encoding = 'big5')
csv_f = csv.reader(f)
next(csv_f)
for row in csv_f:
    if row[2] not in ddict:
        ddict[ row[2] ] = myid;
        myid += 1
        train_data.append([])
    for i in range(3, len(row)):
        if row[i] == 'NR':
            train_data[ ddict[ row[2] ] ].append(0.0)
        else:
            train_data[ ddict[ row[2] ] ].append(float(row[i]))
f.close()
train_data = np.array(train_data)

arrx = [ [] for i in range(len(ddict)) ]

#parse test_X.csv to arrx
with open('test_X.csv', 'r', encoding = 'big5') as csvfile:
    csv_f = csv.reader(csvfile)
    for row in csv_f:
        for i in range(2, len(row)):
            if row[i] == 'NR':
                arrx[ ddict[row[1]] ].append(0.0)
            else:
                arrx[ ddict[row[1]] ].append(float(row[i]))
    arrx = np.array(arrx)

model = LR_model(train_data, arrx, ddict, 0, 0.95, 8e-6)
model.add_data('PM2.5', 9)
model.add_data('PM10', 5)
model.add_data('O3', 3)
model.add_data('RAINFALL', 2)
model.add_data('NO2', 1)
model.training(1100)
model.output_predict()
