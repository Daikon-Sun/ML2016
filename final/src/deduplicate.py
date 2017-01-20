import csv
import re
import pandas as pd
import numpy as np
import sys

train = np.load("expand_train.npy")
print("shape of train is")
print(train.shape)
df = pd.DataFrame(train)
removed_train = df.drop_duplicates()
removed_train = removed_train.as_matrix()
np.save("deduplicate_train.npy",removed_train)
