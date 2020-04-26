import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torch.nn import functional as F
import random

seed=1
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.set_default_dtype(torch.float64)


def load_file(path):
    with open(path, 'rb') as f:
        file = pickle.load(f)
        if type(file) is not np.ndarray:
            file = np.array(file)
        return file


def get_acc(labels,outputs):
    _,predicted = torch.max(outputs.data, 1)
    data_number = y.shape[0]*1.0
    correct_num = (predicted==labels).sum().item()
    accuracy = correct_num/data_num
    return accuracy


X = load_file(r'/Users/sushenzhang/Documents/phd/second_year_code/X_values.pkl')
y = load_file(r'/Users/sushenzhang/Documents/phd/second_year_code/Y_values.pkl')
X = (X-X.min())/(X.max()-X.min())
y= (y-y.min())/(y.max()-y.min())
y = y[:,[0,1,3]]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=12)

X_train = torch.from_numpy(X_train).type(torch.DoubleTensor)
y_train = torch.from_numpy(y_train).type(torch.DoubleTensor)
X_test = torch.from_numpy(X_test).type(torch.DoubleTensor)
y_test = torch.from_numpy(y_test).type(torch.DoubleTensor)

mynet = torch.nn.Sequential(
    torch.nn.Linear(6,10),
    torch.nn.ReLU(),
    torch.nn.Linear(10,3)
    )

optimiser = torch.optim.Adam(mynet.parameters(),lr=0.1)
loss_func = torch.nn.MSELoss(reduction='mean')

losses=[]
for t in range(300):
    out = mynet(X_train)
    loss = loss_func(out,y_train)
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    if t%1 == 0:
        print(loss.item())
        losses.append(loss)

y_pred=mynet(X_test)
print('ypred',y_pred)
print('ytest',y_test)
mse=sum((y_pred-y_test)**2)/y_test.shape[0]
perc_error= sum((y_pred-y_test)/y_test)/y_test.shape[0]
print("MSE",mse)
print('perc_error',perc_error)