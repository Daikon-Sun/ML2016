import pickle
import numpy as np
import csv
import sys
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap, MDS, TSNE
from sklearn.semi_supervised import LabelPropagation
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Activation, merge
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1, l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from keras.utils.visualize_util import plot

def model1():
    data_dim = (3, 32, 32)
    nb_clss = 10

    all_train = pickle.load(open(sys.argv[1]+'all_unlabel.p', 'rb'))
    unlabel_train_x = np.array([ np.array(all_train[i]).reshape(data_dim)\
            for i in range(len(all_train)) ])
    unlabel_train_x = unlabel_train_x.astype('float32')/255

    all_train = pickle.load(open(sys.argv[1]+'all_label.p', 'rb'))
    label_train_x = np.array([ np.array(all_train[i][j]).reshape(data_dim)\
            for i in range(nb_clss) for j in range(len(all_train[i])) ])
    label_train_x = label_train_x.astype('float32')/255

    raw_train_y = np.array([ [i] for i in range(nb_clss)\
            for j in range(len(all_train[i])) ])
    label_train_y = np_utils.to_categorical( raw_train_y )

    #unlabel_train_x = np.vstack((label_train_x, unlabel_train_x))

    all_test = pickle.load(open(sys.argv[1]+'test.p', 'rb'))
    valid_x = np.array([ np.array(all_test['data'][i]).reshape(data_dim)\
            for i in range(len(all_test['data'])) ])
    valid_x = valid_x.astype('float32')/255

    print('adding...')

    model = Sequential()

    model.add(Convolution2D(256, 3, 3, border_mode='same', input_shape=data_dim, W_regularizer=l2(0.0005)))
    model.add(LeakyReLU(alpha=0.5))
    model.add(Convolution2D(128, 3, 3, border_mode='same', W_regularizer=l2(0.0005)))
    model.add(LeakyReLU(alpha=0.4))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(64, 3, 3, border_mode='same', W_regularizer=l2(0.001)))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Convolution2D(64, 3, 3, border_mode='same', W_regularizer=l2(0.001)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Convolution2D(32, 3, 3, border_mode='same', W_regularizer=l2(0.001)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Convolution2D(32, 3, 3, border_mode='same', W_regularizer=l2(0.001)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.35))

    model.add(Flatten())
    model.add(Dense(256, W_regularizer=l2(0.002)))
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.4))
    model.add(Dense(52, W_regularizer=l2(0.003)))
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.5))
    model.add(Dense(nb_clss))
    model.add(Activation('softmax'))

    print('compiling...')
    model.compile(loss='categorical_crossentropy',\
            optimizer='adam',\
            metrics=['accuracy'])

    model.summary()

    plot(model, to_file='model.png')

    model.fit(label_train_x, label_train_y, nb_epoch=100, batch_size=32, verbose=1)

    good_prob=0.96
    print('training...')
    for i in range(1):
        print('iter', i)
        model.fit(label_train_x, label_train_y, nb_epoch=25, batch_size=32, verbose=1)
        prob = model.predict_proba(unlabel_train_x, batch_size=1000)
        ind = np.argwhere(prob>good_prob)
        new_label_train_x = []
        new_label_train_y = []
        if ind.shape[0] == 0:
            continue
        for j in range(ind.shape[0]):
            new_label_train_x.append(unlabel_train_x[ind[j, 0]])
            new_label_train_y.append(ind[j, 1])
        new_label_train_x = np.array(new_label_train_x)
        new_label_train_y = np_utils.to_categorical(np.array(new_label_train_y),\
                nb_classes=10)
        print(new_label_train_x.shape)
        print(new_label_train_y.shape)

        label_train_x = np.vstack((label_train_x, new_label_train_x))
        label_train_y = np.vstack((label_train_y, new_label_train_y))
        print(label_train_x.shape)
        print(label_train_y.shape)

        unlabel_train_x = np.delete(unlabel_train_x, ind[:,0], axis=0)
        print(unlabel_train_x.shape)

    model.save(sys.argv[2])

model1()

#plt.legend(loc='upper left')
#plt.savefig('comparison.png')
