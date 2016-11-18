import pickle
import numpy as np
import csv
import sys
import matplotlib.pyplot as plt
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1, l2
from keras.utils import np_utils
from keras.utils.visualize_util import plot
from sklearn.semi_supervised import LabelPropagation

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

    unlabel_train_x = np.vstack((label_train_x, unlabel_train_x))

    all_test = pickle.load(open(sys.argv[1]+'test.p', 'rb'))
    valid_x = np.array([ np.array(all_test['data'][i]).reshape(data_dim)\
            for i in range(len(all_test['data'])) ])
    valid_x = valid_x.astype('float32')/255

    print('adding...')

    input_img = Input(shape=data_dim)

    x = Convolution2D(64, 3, 3, border_mode='same')(input_img)
    x = LeakyReLU(alpha=1)(x)
    x = MaxPooling2D((2, 2), border_mode='same')(x)

    x = Convolution2D(32, 3, 3, border_mode='same')(x)
    x = LeakyReLU(alpha=0.5)(x)
    x = MaxPooling2D((2, 2), border_mode='same')(x)

    x = Convolution2D(16, 3, 3, border_mode='same')(x)
    x = LeakyReLU(alpha=0.25)(x)
    encoded = MaxPooling2D((2, 2), border_mode='same')(x)

    x = Convolution2D(16, 3, 3, border_mode='same')(encoded)
    x = LeakyReLU(alpha=0.125)(x)
    x = UpSampling2D((2, 2))(x)

    x = Convolution2D(32, 3, 3, border_mode='same')(x)
    x = LeakyReLU(alpha=0.0625)(x)
    x = UpSampling2D((2, 2))(x)

    x = Convolution2D(64, 3, 3, border_mode='same')(x)
    x = LeakyReLU(alpha=0.03125)(x)
    x = UpSampling2D((2, 2))(x)

    x = Convolution2D(3, 3, 3, border_mode='same')(x)
    decoded = Activation('sigmoid')(x)

    autoencoder = Model(input_img, decoded)

    autoencoder.compile(optimizer='rmsprop', loss='mse')

    autoencoder.summary()

    print('training...')
    autoencoder.fit(unlabel_train_x, unlabel_train_x, nb_epoch=3, batch_size=50)

    #decoded_imgs = autoencoder.predict(valid_x)

    '''
    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(1, n+1):
        # display original
        ax = plt.subplot(2, n, i)
        plt.imshow(np.swapaxes(valid_x[i], 0, 2))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + n)
        plt.imshow(np.swapaxes(decoded_imgs[i], 0, 2))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
    '''

    encoder = Model(input=input_img, output=encoded)
    repre = encoder.predict(unlabel_train_x)
    repre = repre.reshape(repre.shape[0],\
            repre.shape[1]*repre.shape[2]*repre.shape[2])

    label_prop_model = LabelPropagation(kernel='knn', n_neighbors=2,\
            alpha=0.999, n_jobs=-1, max_iter=1)
    tmp = -np.ones((unlabel_train_x.shape[0]-raw_train_y.shape[0], 1)).astype('int')
    final_labels = np.vstack((raw_train_y,tmp))
    final_labels = np.transpose(final_labels).tolist()[0]

    label_prop_model.fit(repre, final_labels)
    final_labels = label_prop_model.transduction_

    model = Sequential()

    print('adding...')

    model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=data_dim,\
            W_regularizer=l2(0.00003)))
    model.add(LeakyReLU(alpha=0.5))
    model.add(Convolution2D(64, 3, 3, border_mode='same', W_regularizer=l2(0.00004)))
    model.add(LeakyReLU(alpha=0.4))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Convolution2D(32, 3, 3, border_mode='same', W_regularizer=l2(0.00005)))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Convolution2D(32, 3, 3, border_mode='same', W_regularizer=l2(0.00006)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(16, 3, 3, border_mode='same', W_regularizer=l2(0.00007)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Convolution2D(16, 3, 3, border_mode='same', W_regularizer=l2(0.00008)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(256, W_regularizer=l2(0.00009)))
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.4))
    model.add(Dense(52, W_regularizer=l2(0.0001)))
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.5))
    model.add(Dense(nb_clss))
    model.add(Activation('softmax'))

    print('compiling...')
    model.compile(loss='categorical_crossentropy',\
            optimizer='adam',\
            metrics=['accuracy'])

    final_labels = np_utils.to_categorical(final_labels)

    #np.random.seed(0)
    #np.random.shuffle(unlabel_train_x)
    #np.random.seed(0)
    #np.random.shuffle(final_labels)

    hist = model.fit(unlabel_train_x, final_labels, nb_epoch=10, batch_size=50)

    #plt.plot(hist.history['val_acc'], label='val_acc')
    #plt.plot(hist.history['acc'], label='acc')

    model.save(sys.argv[2])

model1()
