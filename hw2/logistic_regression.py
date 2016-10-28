import csv
import numpy as np
import sys
from random import randint

all_test_data = []
with open('spam_data/spam_test.csv', 'r') as csvfile:
    csvf = csv.reader(csvfile)
    for row in csvf:
        all_test_data.append([])
        rid = int(row[0])-1
        for i in range(1, len(row)):
            all_test_data[rid].append(float(row[i]))
all_test_data = np.array(all_test_data)

class LR_model:
    def __init__(self, train_data, valid_data, train_ans, valid_ans,
                 myrho, myepsilon):

        self.all_trn_dat = np.array(train_data)
        self.all_vld_dat = np.array(valid_data)
        self.trn_ans = np.array(train_ans)
        self.vld_ans = np.array(valid_ans)
        self.rho = myrho
        self.eps = myepsilon
        self.trn_len = self.all_trn_dat.shape[0]
        self.vld_len = self.all_vld_dat.shape[0]
        self.trn_dat = []
        self.vld_dat = []
        self.idx_list = []

    def add_data(self, idx):
        self.trn_dat.append(self.all_trn_dat[:, idx])
        if self.vld_len > 0:
            self.vld_dat.append(self.all_vld_dat[:, idx])
        self.idx_list.append(idx)

    def init(self):
        '''
        initialization (must call this function before training)
        '''
        self.trn_dat.append([ 1. for i in range(self.trn_len) ])    #bias term
        self.vld_dat.append([ 1. for i in range(self.vld_len) ])    #bias term
        self.trn_dat = np.transpose(np.array(self.trn_dat))
        self.vld_dat = np.transpose(np.array(self.vld_dat))
        self.w = np.zeros((self.trn_dat.shape[1], 1))
        self.ada = np.zeros((self.trn_dat.shape[1], 1))
        self.ada_d = np.zeros((self.trn_dat.shape[1], 1))
        self.ada_g = np.zeros((self.trn_dat.shape[1], 1))

    def train(self, btch_cnt, iternum, lr, choice = 0):
        for i in range(1, iternum+1):

            #randomize training set
            self.trn_dat = np.append(self.trn_dat, self.trn_ans, axis = 1)
            np.random.shuffle(self.trn_dat)
            self.trn_ans = self.trn_dat[:,-1:]
            self.trn_dat = np.delete(self.trn_dat, -1, axis = 1)

            #training
            btch_sz = int(self.trn_len/btch_cnt)
            for j in range(btch_cnt):
                head = j*btch_sz
                tail = (j+1)*btch_sz
                res = np.transpose(self.trn_ans[head:tail,:]-(1/\
                        (1+np.exp(-np.dot(self.trn_dat[head:tail,:], self.w)))))
                gra = -np.transpose(np.dot(res, self.trn_dat[head:tail,:]))
                if choice == 0: #adadelta:
                    self.ada_g = self.ada_g*self.rho+(1.-self.rho)*(gra**2)
                    dw = -gra*(self.ada_d+self.eps)**0.5/(self.ada_g+self.eps)**0.5
                    self.ada_d = self.rho*self.ada_d+(1.-self.rho)*(dw**2)
                    self.w += dw
                elif choice == 1: #adagrad:
                    self.ada += (gra**2)
                    self.w -= (lr/self.w.shape[0])*gra/(self.ada**0.5)
                elif choice == 2: #simple:
                    self.w -= (lr/self.w.shape[0])*gra
                else:
                    assert False
            if i%10 == 0:
                print(self.evaluate(self.vld_dat, self.vld_ans))

    def evaluate(self, dat, ans):
        cor_cnt = 0.0
        res = np.dot(dat, self.w)
        assert res.shape[0] == ans.shape[0]
        for i in range(res.shape[0]):
            if res[i, 0] >= 0 and ans[i, 0] == 1:
                cor_cnt += 1
            elif res[i, 0] < 0 and ans[i, 0] == 0:
                cor_cnt += 1
        return cor_cnt/res.shape[0]

    def output_parameters(self, file_name):
        with open(file_name, 'w') as csvfile:
            wrtr = csv.writer(csvfile)
            list_len = len(self.idx_list)
            for i in range(list_len):
                wrtr.writerow([self.idx_list[i], self.w[i, 0]])
            wrtr.writerow([10000, self.w[list_len, 0]])

train_data = []
valid_data = []
train_ans = []
valid_ans = []
f = open(sys.argv[1], 'r')
csv_f = csv.reader(f)
for row in csv_f:
    train_data.append([])
    train_ans.append([])
    for i in range(1, len(row)-1):
        train_data[int(row[0])-1].append(float(row[i]))
    train_ans[int(row[0])-1].append(int(row[len(row)-1]))
f.close()

for i in range(700):
    x = randint(0, 4001-i-1)
    valid_data.append(train_data.pop(x))
    valid_ans.append(train_ans.pop(x))

iter_num = 90#000#0
learning_rate = 1.6
btch_cnt = 10
test_model=LR_model(train_data,valid_data,train_ans,valid_ans,0.95,1e-7)
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
test_model.init()
test_model.train(btch_cnt, iter_num, learning_rate)
test_model.output_parameters(sys.argv[2])
