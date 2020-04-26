"""
Find the correlation matrix
"""
import numpy as np
import pandas as pd
import pickle
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def load_file(path):
    with open(path, 'rb') as f:

        file = pickle.load(f)

        if type(file) is not np.ndarray:
            file = np.array(file)

        return file

X = load_file(r'/Users/sushenzhang/Documents/phd/second_year_code/X_values.pkl')
y = load_file(r'/Users/sushenzhang/Documents/phd/second_year_code/Y_values.pkl')
X = (X-X.min())/(X.max()-X.min())
y= (y-y.min())/(y.max()-y.min())

y = y[[0,1,3]]
df=pd.DataFrame(y)
corr=df.corr()
corr.style.background_gradient(cmap='coolwarm')
fig, ax = plt.subplots(figsize=(3, 3))
ax.matshow(corr)
plt.xticks(range(len(corr.columns)), corr.columns)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.show()