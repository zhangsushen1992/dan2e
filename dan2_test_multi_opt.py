'''
Optimisation scheme
'''
"""
Multitask testing
"""

import dan2_multi_opt as dan2
import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

TOLERANCE=0.1
TOLERANCE_ts_1=0.1
TOLERANCE_ts_2=0.1
TOLERANCE_ts_3=0.1

def load_file(path):
    with open(path, 'rb') as f:

        file = pickle.load(f)

        if type(file) is not np.ndarray:
            file = np.array(file)

        return file


def test_fit_and_predict(training_preds, testing_preds):
    return np.array_equal(training_preds, testing_preds)


def main(X_train, X_test, y_train, y_test, depth):
    clf = dan2.DAN2Regressor(depth=depth)
    tr_pred, tr_mse = clf.fit(X_train, y_train)
    return tr_pred, tr_mse, clf


def multi(X, y, head_depth, head_number, prev_f_k, model):
    mse, f_k = model.multihead(X, y, head_depth, head_number, prev_f_k)
    return mse, f_k


def test(X_test, y_test, head_number,model):
    y_pred = model.predict(X_test,head_number)
    mse = np.sum((y_pred - y_test)**2) / X_test.shape[0]
    perc_error=np.sum(np.abs((y_pred-y_test)/y_test))/y_test.shape[0]
    return mse,perc_error

if __name__ == '__main__':
    
    X = load_file(r'/Users/sushenzhang/Documents/phd/second_year_code/X_values.pkl')
    y = load_file(r'/Users/sushenzhang/Documents/phd/second_year_code/Y_values.pkl')
    X = (X-X.min())/(X.max()-X.min())
    y= (y-y.min())/(y.max()-y.min())


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.30, random_state=42)
  
    y_train_1 = y_train[:,0]
    y_test_1 = y_test[:,0]
    y_train_1 = y_train_1.reshape(len(y_train), 1)
    y_test_1 = y_test_1.reshape(len(y_test),1)
  
    y_train_2 = y_train[:,1]
    y_test_2 = y_test[:,1]
    y_train_2 = y_train_2.reshape(len(y_train), 1)
    y_test_2 = y_test_2.reshape(len(y_test),1)

    y_train_3 = y_train [:,3]
    y_test_3 =y_test[:,3]

    y_train_3 = y_train_3.reshape(len(y_train), 1)
    y_test_3 = y_test_3.reshape(len(y_test),1)

    
    y_train_total = np.vstack((y_train_1, y_train_2, y_train_3))
    y_test_total = np.vstack((y_test_1, y_test_2, y_test_3))
    X_train_total = np.vstack((X_train, X_train, X_train))
    X_test_total = np.vstack((X_test, X_test, X_test))
    
    sse_list = []
    sse=1
    sse_prev=0.01
    sse_ts=1
    sse_prev_ts=0.01
    xi_sh=0
    xi_ts=0
    f = dan2.add_layer(X_train_total,y_train_total,isFirstLayer=True)
    while sse>TOLERANCE:
        while (sse - sse_prev)/sse_prev > TOLERANCE_1:
            sse_prev = sse
            f = dan2.add_layer(X_train_total, y_train_total, False, f)
            sse = np.sum(f-y_train_total)**2
            sse_list.append(sse)
            xi_sh = xi_sh+1
            print('Number of shared layers is', xi_sh)
        while (sse_1-sse_prev_1)/sse_prev_1>TOLERANCE_ts_1:
            sse_prev_1=sse_1
            f_ts_1 = dan2.add_layer_multihead(X_train, y_trian_1,0, f)
            sse_1 = np.sum(f_ts_1-y_train_1)**2
            xi_1 = xi_1+1
            print('Number of layers for Task 1 is', xi_1)
        while (sse_2-sse_prev_2)/sse_prev_2>TOLERANCE_ts_2:
            sse_prev_2=sse_1
            f_ts_2 = dan2.add_layer_multihead(X_train, y_trian_2,0, f)
            sse_2 = np.sum(f_ts_2-y_train_2)**2
            xi_2 = xi_2+1
            print('Number of layers for Task 2 is', xi_2)
        while (sse_3-sse_prev_3)/sse_prev_3>TOLERANCE_ts_3:
            sse_prev_3=sse_3
            f_ts_3 = dan2.add_layer_multihead(X_train, y_trian_3,0, f)
            sse_3 = np.sum(f_ts_3-y_train_3)**2
            xi_3 = xi_3+1
            print('Number of layers for Task 3 is', xi_3)
    plt.plot(sse_list)


    # mse_list.append((mse_1,mse_2,mse_3))
    # tst_mse_1, perc_error_1 = test(X_test_total, y_test_total, 0, clf)
    # tst_mse_2, perc_error_2 = test(X_test_total, y_test_total, 1, clf)
    # tst_mse_3, perc_error_3 = test(X_test_total, y_test_total, 2,clf)
    # print("training mse", mse_1, mse_2, mse_3)
    # print("MSE",  tst_mse_1,tst_mse_2,tst_mse_3)
    # print("APE ", perc_error_1, perc_error_2, perc_error_3)
