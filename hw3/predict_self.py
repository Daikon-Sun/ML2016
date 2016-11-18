import pickle
import numpy as np
import sys
import csv
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.models import load_model

all_test = pickle.load(open(sys.argv[1]+'test.p', 'rb'))
test_x = np.array([ np.array(all_test['data'][i]).reshape((3, 32, 32))\
        for i in range(len(all_test['data'])) ])
test_x = test_x.astype('float32')/255

model = load_model(sys.argv[2])
model.summary()

prd = model.predict_classes(test_x, batch_size=1000)
with open(sys.argv[3], 'w') as csvfile:
    csv_f = csv.writer(csvfile)
    csv_f.writerow(['ID']+['class'])
    for i in range(prd.shape[0]):
        csv_f.writerow([i]+[prd[i, ]])
