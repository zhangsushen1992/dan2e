"""
Optimisation scheme
"""
"""
Multitask testing
"""

import dan2_multi_opt as dan2
import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time

TOLERANCE=0.07
TOLERANCE_1=0.001
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


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
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
    
    sse_train_list = []
    sse_val_list=[]
    sse_val_list_1, sse_val_list_2, sse_val_list_3=[],[],[]
    sse=1
    sse_val=1
    sse_prev=0.01
    sse_1=1
    sse_prev_1=0.01
    sse_2=1
    sse_prev_2=0.01
    sse_3=1
    sse_prev_3=0.01
    xi_sh,xi_1,xi_2,xi_3=0,0,0,0
    sse_val, sse_val_1, sse_val_2, sse_val_3=1,1,1,1
  
    start_time =time.time()
    clf = dan2.DAN2Regressor()
    f = clf.add_layer(X=X_train_total,y=y_train_total,isFirstLayer=True,f_k=0)
    while sse>TOLERANCE:
        print('sse is',sse)
        while np.abs(sse_prev - sse_val )/sse_prev > TOLERANCE_1:
            sse_prev = sse_val
            f = clf.add_layer(X_train_total, y_train_total, False, f)
            sse = np.sum((f-y_train_total)**2)
            sse_train_list.append(sse)
            xi_sh = xi_sh+1
            f_val = clf.predict_shared(X_test_total)
            sse_val = np.sum((f_val-y_test_total)**2)/y_test_total.shape[0]
            sse_val_list.append(sse_val)
            print('Number of shared layers is', xi_sh)
            print('new sse is',sse)
            print('sseval is',(sse_val_1-sse_prev/sse_prev))
        while np.abs(sse_val_1-sse_prev_1)/sse_prev_1>TOLERANCE_ts_1:
            sse_prev_1=sse_val_1
            f_ts_1 = clf.add_layer_multihead(X_train, y_train_1,0, f[0:1654])
            sse_1 = np.sum((f_ts_1-y_train_1)**2)
            xi_1 = xi_1+1
            f_ts_val_1 = clf.predict(X_test,0)
            sse_val_1 = np.sum((f_ts_val_1-y_test_1)**2)/y_test_1.shape[0]
            sse_val_list_1.append(sse_val_1)
            print('Number of layers for Task 1 is', xi_1)
            print('sse1 is',sse_1)
            print('sseval is',sse_val_1)
        while np.abs(sse_val_2-sse_prev_2)/sse_prev_2>TOLERANCE_ts_2:
            if sse_val_2-sse_prev_2>5:
                break
            sse_prev_2=sse_val_2
            f_ts_2 = clf.add_layer_multihead(X_train, y_train_2,1, f[1654:3308])
            sse_2 = np.sum((f_ts_2-y_train_2)**2)
            xi_2 = xi_2+1
            f_ts_val_2 = clf.predict(X_test,1)
            sse_val_2 = np.sum((f_ts_val_2-y_test_2)**2)/y_test_2.shape[0]
            sse_val_list_2.append(sse_val_2)
            print('Number of layers for Task 2 is', xi_2)
            print('sse2 is',sse_2)
            print('sseval is',sse_val_2)
        while np.abs(sse_val_3-sse_prev_3)/sse_prev_3>TOLERANCE_ts_3:
            if sse_val_3-sse_prev_3>5:
                break
            sse_prev_3=sse_val_3
            f_ts_3 = clf.add_layer_multihead(X_train, y_train_3,2, f[3308:])
            sse_3 = np.sum((f_ts_3-y_train_3)**2)
            xi_3 = xi_3+1
            f_ts_val_3 = clf.predict(X_test,2)
            sse_val_3 = np.sum((f_ts_val_3-y_test_3)**2)/y_test_3.shape[0]
            sse_val_list_3.append(sse_val_3)
            print('Number of layers for Task 3 is', xi_3)
            print('sse3 is',sse_3)
            print('sseval is',sse_val_3)
    end_time = time.time()
    list_1 = sse_val_list+sse_val_list_1
    list_2 = sse_val_list+sse_val_list_2
    list_3 = sse_val_list+sse_val_list_3
    print('list 1',list_1)
    print('list 2',list_2)
    print('list 3',list_3)

    print('Shared Layers:',xi_sh)
    print('Task 1 Layer:', xi_1)
    print('Task 2 Layer', xi_2)
    print('Task 3 Layer', xi_3)
    print('Training Loss',sse, sse_1, sse_2, sse_3)
    print('Validation Loss', sse_val, sse_val_1, sse_val_2, sse_val_3)
    print('Average training SSE', sse_1+sse_2+sse_3)
    print('Average validation MSE', (sse_val_1+sse_val_2+sse_val_3)/3)
    print('Total CPU time is', end_time- start_time)
  
    # fig, ax1 = plt.subplots()
    # ax1.plot(list_1,label='Task 1',c='m')
    # ax1.set_xlabel('The Number of Layers')
    # ax2=ax1.twinx()
    # ax2.plot(list_2,label='Task 2',c='r')
    # ax3=ax1.twinx()
    # ax3.plot(list_3,label='Task 3',c='b')
    # # ax4=ax1.twinx()
    # # ax4.plot(sse_val_list,label='Shared Layers',c='g')
    # fig.tight_layout()
    # ax1.legend()
    # ax2.legend()
    # ax3.legend()
    plt.plot(list_1,label='Task 1',c='m')
    plt.plot(list_2,label='Task 2',c='r')
    plt.plot(list_3,label='Task 3',c='b')
    plt.xlabel('Number of Iterations')
    plt.legend()
    plt.ylabel('The Validation Loss')
    plt.title('Architectural Optimisation using Validation Loss')
    plt.show()


    # mse_list.append((mse_1,mse_2,mse_3))
    # tst_mse_1, perc_error_1 = test(X_test_total, y_test_total, 0, clf)
    # tst_mse_2, perc_error_2 = test(X_test_total, y_test_total, 1, clf)
    # tst_mse_3, perc_error_3 = test(X_test_total, y_test_total, 2,clf)
    # print("training mse", mse_1, mse_2, mse_3)
    # print("MSE",  tst_mse_1,tst_mse_2,tst_mse_3)
    # print("APE ", perc_error_1, perc_error_2, perc_error_3)
