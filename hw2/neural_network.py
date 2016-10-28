import csv
import numpy as np
import sys
import random

class adagrad:
    def __init__(self, lr):
        self.lr = lr

    def info(self):
        return 'learning rate:{:.1e}'.format(self.lr)

    def init(self, shapes):
        self.num_layers = len(shapes)
        self.ada_b = [np.zeros((y, 1)) for y in shapes[1:]]
        self.ada_w = [np.zeros((y, x))\
                for x, y in zip(shapes[:-1], shapes[1:])]

    def update(self, w, b, gra_w, gra_b):
        for i in range(self.num_layers-1):
            self.ada_w[i] += gra_w[i]**2
            w[i] -= self.lr*gra_w[i]/np.sqrt(self.ada_w[i])
            self.ada_b[i] += gra_b[i]**2
            b[i] -= self.lr*gra_b[i]/np.sqrt(self.ada_b[i])
class adadelta:
    def __init__(self, rho, eps):
        self.rho = rho
        self.eps = eps

    def info(self):
        return 'rho:{:.1e}, eps:{:.1e}'.format(self.rho, self.eps)

    def init(self, shapes):
        self.num_layers = len(shapes)
        self.ada_gb = [np.zeros((y, 1)) for y in shapes[1:]]
        self.ada_db = [np.zeros((y, 1)) for y in shapes[1:]]
        self.ada_gw = [np.zeros((y, x))\
                for x, y in zip(shapes[:-1], shapes[1:])]
        self.ada_dw = [np.zeros((y, x))\
                for x, y in zip(shapes[:-1], shapes[1:])]

    def update(self, w, b, gra_w, gra_b):
        for i in range(self.num_layers-1):
            self.ada_gw[i] = self.ada_gw[i]*self.rho+(1-self.rho)*gra_w[i]**2
            adw = -gra_w[i]*(self.ada_dw[i]+self.eps)**0.5/\
                    (self.ada_gw[i]+self.eps)**0.5
            self.ada_dw[i] = self.rho*self.ada_dw[i]+(1-self.rho)*adw**2
            w[i] += adw
            self.ada_gb[i] = self.ada_gb[i]*self.rho+(1-self.rho)*gra_b[i]**2
            adb = -gra_b[i]*(self.ada_db[i]+self.eps)**0.5/\
                    (self.ada_gb[i]+self.eps)**0.5
            self.ada_db[i] = self.rho*self.ada_db[i]+(1-self.rho)*adb**2
            b[i] += adb
class RMSprop:
    def __init__(self, lr, dr, eps):
        self.lr = lr
        self.dr = dr
        self.eps = eps

    def info(self):
        return 'lr:{:.1e}, dr:{}, eps:{:.1e}'.format(self.lr, self.dr, self.eps)

    def init(self, shapes):
        self.num_layers = len(shapes)
        self.cache_b = [np.zeros((y, 1)) for y in shapes[1:]]
        self.cache_w = [np.zeros((y, x))\
                for x, y in zip(shapes[:-1], shapes[1:])]

    def update(self, w, b, gra_w, gra_b):
        for i in range(self.num_layers-1):
            self.cache_w[i] = self.dr*self.cache_w[i]+(1-self.dr)*gra_w[i]**2
            w[i] -= self.lr*gra_w[i]/(np.sqrt(self.cache_w[i])+self.eps)
            self.cache_b[i] = self.dr*self.cache_b[i]+(1-self.dr)*gra_b[i]**2
            b[i] -= self.lr*gra_b[i]/(np.sqrt(self.cache_b[i])+self.eps)
class adam:
    def __init__(self, lr, beta1, beta2, eps):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def info(self):
        return 'lr:{:.1e}, beta1:{}, beta2:{}, eps:{:.1e}'.\
                format(self.lr, self.beta1, self.beta2, self.eps)

    def init(self, shapes):
        self.num_layers = len(shapes)
        self.mb = [np.zeros((y, 1)) for y in shapes[1:]]
        self.vb = [np.zeros((y, 1)) for y in shapes[1:]]
        self.mw = [np.zeros((y, x))\
                for x, y in zip(shapes[:-1], shapes[1:])]
        self.vw = [np.zeros((y, x))\
                for x, y in zip(shapes[:-1], shapes[1:])]

    def update(self, w, b, gra_w, gra_b):
        for i in range(self.num_layers-1):
            self.mw[i] = self.beta1*self.mw[i]+(1-self.beta1)*gra_w[i]
            self.vw[i] = self.beta2*self.vw[i]+(1-self.beta2)*gra_w[i]**2
            w[i] -= self.lr*self.mw[i]/(np.sqrt(self.vw[i])+self.eps)
            self.mb[i] = self.beta1*self.mb[i]+(1-self.beta1)*gra_b[i]
            self.vb[i] = self.beta2*self.vb[i]+(1-self.beta2)*gra_b[i]**2
            b[i] -= self.lr*self.mb[i]/(np.sqrt(self.vb[i])+self.eps)
class nesterov_momentum:
    def __init__(self, lr, mu):
        self.lr = lr
        self.mu = mu

    def info(self):
        return 'lr:{:.1e}, mu:{}'.format(self.lr, self.mu)

    def init(self, shapes):
        self.num_layers = len(shapes)
        self.vb = [np.zeros((y, 1)) for y in shapes[1:]]
        self.vw = [np.zeros((y, x))\
                for x, y in zip(shapes[:-1], shapes[1:])]

    def update(self, w, b, gra_w, gra_b):
        for i in range(self.num_layers-1):
            tmp_vw = self.vw[i]
            self.vw[i] = self.mu*self.vw[i]-self.lr*gra_w[i]
            w[i] -= self.mu*tmp_vw+(1+self.mu)*self.vw[i]
            tmp_vb = self.vb[i]
            self.vb[i] = self.mu*self.vb[i]-self.lr*gra_b[i]
            b[i] -= self.mu*tmp_vb+(1+self.mu)*self.vb[i]

class sigmoid:
    @staticmethod
    def f(x):
        return 1.0/(1.0+np.exp(-x))
    @staticmethod
    def d_f(x):
        return np.exp(-x)/(1.0+np.exp(-x))
    @staticmethod
    def info():
        return 'sigmoid'
    @staticmethod
    def idx():
        return [[0]]
class tanh:
    @staticmethod
    def f(x):
        return np.tanh(x)
    @staticmethod
    def d_f(x):
        return 1-np.tanh(x)**2
    @staticmethod
    def info():
        return 'tanh'
    @staticmethod
    def idx():
        return [[1]]
class arctan:
    @staticmethod
    def f(x):
        return np.arctan(x)
    @staticmethod
    def d_f(x):
        return 1/(1+x**2)
    @staticmethod
    def info():
        return 'atctan'
    @staticmethod
    def idx():
        return [[2]]
class relu:
    @staticmethod
    def f(x):
        return np.maximum(0, x)
    @staticmethod
    def d_f(x):
        return 1.*(x>0)
    @staticmethod
    def info():
        return 'relu'
    @staticmethod
    def idx():
        return [[3]]
class leaky_relu:
    @staticmethod
    def f(x):
        return alpha*x*(x<=0)+np.maximum(0, x)
    @staticmethod
    def d_f(x):
        return alpha*(x<=0)+1.*(x>0)
    @staticmethod
    def info():
        return 'leaky relu: {}'.format(alpha)
    @staticmethod
    def idx():
        return [[4, alpha]]

class neural_network:
    def __init__(self, all_train_data, all_valid_data, method, fn):

        self.all_trn_dat = all_train_data
        self.all_vld_dat = all_valid_data
        self.trn_len = len(self.all_trn_dat)
        self.vld_len = len(self.all_vld_dat)
        self.trn_dat = [ [ [], self.all_trn_dat[i][1] ]\
                for i in range(self.trn_len) ]
        self.vld_dat = [ [ [], self.all_vld_dat[i][1] ]\
                for i in range(self.vld_len) ]
        self.idx_list = []
        self.method = method
        self.fn = fn

    def add_data(self, idx):
        self.idx_list.append(idx)
        for i in range(self.trn_len):
            self.trn_dat[i][0].append(self.all_trn_dat[i][0][idx, :])
        for i in range(self.vld_len):
            self.vld_dat[i][0].append(self.all_vld_dat[i][0][idx, :])

    def init(self, shapes, dop):
        '''
        initialization (must call this function before training)
        '''
        assert len(self.trn_dat) == self.trn_len
        for i in range(self.trn_len):
            self.trn_dat[i][0] = np.array(self.trn_dat[i][0])
            self.trn_dat[i][1] = np.array(self.trn_dat[i][1])
        for i in range(self.vld_len):
            self.vld_dat[i][0] = np.array(self.trn_dat[i][0])
            self.vld_dat[i][1] = np.array(self.trn_dat[i][1])
        self.shapes = [len(self.idx_list)]+shapes
        self.dop = dop
        self.num_layers = len(self.shapes)
        #self.trn_dat = np.array(self.trn_dat)
        self.b = [np.random.randn(y, 1) for y in self.shapes[1:]]
        self.w = [np.random.randn(y, x)/np.sqrt(x) for x, y in\
                zip(self.shapes[:-1], self.shapes[1:])]

        self.method.init(self.shapes)

    def backprop(self, x, y, drop):
        actvt = x*drop[0]
        actvts = [x]
        zs = []
        for w, b, dp in zip(self.w, self.b, drop[1:]):
            zs.append(np.dot(w, actvts[-1])+b)
            actvts.append(self.fn.f(zs[-1])*dp)

        dw = [np.zeros(w.shape) for w in self.w]
        db = [np.zeros(b.shape) for b in self.b]
        delta = (actvts[-1]-y)
        db[-1] = delta
        dw[-1] = np.dot(delta, np.transpose(actvts[-2]))
        for l in range(2, self.num_layers):
            delta = np.dot(np.transpose(self.w[-l+1]), delta)\
                    *self.fn.d_f(zs[-l])*drop[-l]
            db[-l] = delta
            dw[-l] = np.dot(delta, np.transpose(actvts[-l-1]))
        return dw, db

    def forward(self, a):
        for w, b in zip(self.w, self.b):
            a = self.fn.f(np.dot(w, a)+b)
        return a

    def update(self, minibatch, lmbd):
        gra_w = [np.zeros(w.shape) for w in self.w]
        gra_b = [np.zeros(b.shape) for b in self.b]

        drop = []
        for i in range(self.num_layers-1):
            drop.append(np.random.choice([0, 1], (self.shapes[i], 1)\
                    ,p=[self.dop, 1-self.dop])/(1-self.dop))
        drop.append(np.ones((self.shapes[-1], 1))/(1-self.dop))
        #print(drop)
        dlen = len(minibatch)
        for x, y in minibatch:
            assert x.shape[1] == y.shape[1]
            dw, db = self.backprop(x, y, drop)
            gra_w = [nw+ndw for nw, ndw in zip(gra_w, dw)]
            gra_b = [nb+ndb for nb, ndb in zip(gra_b, db)]

        #print(gra_b)

        if lmbd > 0:
            for i in range(self.num_layers-1):
                gra_w[i] += self.w[i]*lmbd

        self.method.update(self.w, self.b, gra_w, gra_b)

    def train(self, batch_cnt, iternum, lmbd):
        for i in range(1, iternum+1):

            btch_sz = int(self.trn_len/btch_cnt)
            random.shuffle(self.trn_dat)
            for j in range(btch_cnt):
                self.update(self.trn_dat[j*btch_sz:(j+1)*btch_sz], lmbd)
            if i%20 == 0:
                print('train acc:{}'.format(self.evaluate(True)))
                print('valid acc:{}'.format(self.evaluate(False)))
                print('lmbd:{:.1e}, method:{}, idx_list:{}'.\
                        format(lmbd, self.method.info(), len(self.idx_list)))
                print('-----------epoch:{}, shapes:{}, btch_cnt:{}, actvt fn:{}'.\
                        format(i, self.shapes, btch_cnt, self.fn.info()))

    def evaluate(self, train_set):
        if train_set:
            test_res = [(np.argmax(self.forward(x)), np.argmax(y))\
                    for (x, y) in self.trn_dat]
            return sum(int(x==y) for x, y in test_res)/self.trn_len
        else:
            test_res = [(np.argmax(self.forward(x)), np.argmax(y))\
                    for (x, y) in self.vld_dat]
            return sum(int(x==y) for x, y in test_res)/self.vld_len

    def save_parameters(self, file_name):
        f = open(file_name, 'wb')
        np.save(f, data_mean, False)
        np.save(f, data_std, False)
        np.save(f, np.array(self.idx_list), False)
        np.save(f, np.array(self.shapes), False)
        np.save(f, np.array(self.fn.idx()))
        for i in range(self.num_layers-1):
            np.save(f, self.w[i], False)
            np.save(f, self.b[i], False)
        f.close()

data_list = [ [] for i in range(57) ]
train_data = []
valid_data = []
train_ans = []
valid_ans = []
f = open(sys.argv[1], 'r')
csv_f = csv.reader(f)
for row in csv_f:
    train_data.append([])
    tmp_dat = []
    for i in range(1, len(row)-1):
        data_list[i-1].append(float(row[i]))
        tmp_dat.append([float(row[i])])
    train_data[-1].append(np.array(tmp_dat))
    if int(row[-1]) == 1:
        train_data[-1].append(np.array([[0],[1]]))
    else:
        train_data[-1].append(np.array([[1],[0]]))
f.close()

data_mean = []
data_std = []
for i in range(57):
    data_list[i] = np.array(data_list[i])
    data_mean.append([np.mean(data_list[i])])
    data_std.append([np.std(data_list[i])])

data_mean = np.array(data_mean)
data_std = np.array(data_std)
for i in range(len(train_data)):
    for j in range(57):
        train_data[i][0][j, 0]=\
                (train_data[i][0][j, 0]-data_mean[j, 0])/data_std[j, 0]

for i in range(700):
    x = random.randint(0, 4001-i-1)
    valid_data.append(train_data.pop(x))

iter_num = 100#000
lmbd = 0.001
shapes = [57, 2]
btch_cnt = 10
dop = 0.05

alpha = 0.009
fn = leaky_relu()

#method = adagrad(learning_rate)
#method = adagrad(1e-2)

#method = adadelta(rho, eps)
method = adadelta(0.95, 1e-6)

#method = RMSprop(learning_rate, decay_rate, eps)
#method = RMSprop(2e-3, 0.99, 1e-6)

#method = adam(lr, beta1, beta2, eps)
#method = adam(6e-4, 0.9, 0.999, 1e-8)

#method = nesterov_momentum(lr, mu)
#method = nesterov_momentum(1e-4, 0.9)

test_model = neural_network(train_data, valid_data, method, fn)
test_model.add_data(0)# make
test_model.add_data(1)# address x
test_model.add_data(2)# all x
test_model.add_data(3)# 3d
test_model.add_data(4)# our
test_model.add_data(5)# over //good
test_model.add_data(6)# remove x //good
test_model.add_data(7)# internet
test_model.add_data(8)# order x //good
test_model.add_data(9)# mail x //good
test_model.add_data(10)# receive x
test_model.add_data(11)# will x
test_model.add_data(12)# people
test_model.add_data(13)# report x
test_model.add_data(14)# addresses x
test_model.add_data(15)# free x //good
test_model.add_data(16)# business x
test_model.add_data(17)# email x
test_model.add_data(18)# you x
test_model.add_data(19)# credit
test_model.add_data(20)# your x //good
test_model.add_data(21)# font
test_model.add_data(22)# 000 x
test_model.add_data(23)# money x
test_model.add_data(24)# hp //good
test_model.add_data(25)# hpl x
test_model.add_data(26)# george //good
test_model.add_data(27)# 650
test_model.add_data(28)# lab
test_model.add_data(29)# labs
test_model.add_data(30)# telnet
test_model.add_data(31)# 857
test_model.add_data(32)# data
test_model.add_data(33)# 415
test_model.add_data(34)# 85
test_model.add_data(35)# echnology
test_model.add_data(36)# 1999 x
test_model.add_data(37)# parts x
test_model.add_data(38)# pm
test_model.add_data(39)# direct
test_model.add_data(40)# cs
test_model.add_data(41)# meeting //good
test_model.add_data(42)# original
test_model.add_data(43)# project
test_model.add_data(44)# re x
test_model.add_data(45)# edu x
test_model.add_data(46)# table
test_model.add_data(47)# conference
test_model.add_data(48)# char_freq_; x
test_model.add_data(49)# char_freq_(
test_model.add_data(50)# char_freq_[
test_model.add_data(51)# char_freq_! x //good
test_model.add_data(52)# char_freq_$
test_model.add_data(53)# char_freq_# x
test_model.add_data(54)# capital_run_length_average x
test_model.add_data(55)# capital_run_length_longest x
test_model.add_data(56)# capital_run_length_total x
test_model.init(shapes, dop)
test_model.train(btch_cnt, iter_num, lmbd)
test_model.save_parameters(sys.argv[2])
