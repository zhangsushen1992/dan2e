"""
Optimisation scheme
"""
"""
Multitask testing
"""

import dan2_multi_opt as dan2
import numpy as np
import pandas as pd
import pickle
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time

TOLERANCE=0.01
TOLERANCE_1=0.1
TOLERANCE_ts_1=0.3
TOLERANCE_ts_2=0.3
TOLERANCE_ts_3=0.3
TOLERANCE_ts_4=0.3

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
    
   
    df = pd.read_csv('./full.csv')
    X = df[['dep','octanol','octanoic','pentanol','temperature','humidity']]
    y = df[['average_speed','average_number_of_droplets_last_second','max_average_single_droplet_speed','average_number_of_droplets']]
    X = X.to_numpy()
    y = y.to_numpy()
    X = (X-X.min())/(X.max()-X.min())
    y = (y-y.min())/(y.max()-y.min())


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.30, random_state=42)
  
    y_train_1 = y_train[:,0]
    y_test_1 = y_test[:,0]
    y_train_1 = y_train_1.reshape(len(y_train), 1)
    y_test_1 = y_test_1.reshape(len(y_test),1)
  
    y_train_2 = y_train[:,2]
    y_test_2 = y_test[:,2]
    y_train_2 = y_train_2.reshape(len(y_train), 1)
    y_test_2 = y_test_2.reshape(len(y_test),1)
  
    y_train_3 = y_train[:,3]
    y_test_3 = y_test[:,3]
    y_train_3 = y_train_3.reshape(len(y_train), 1)
    y_test_3 = y_test_3.reshape(len(y_test),1)

    y_train_4 = y_train [:,1]
    y_test_4 =y_test[:,1]
    y_train_4 = y_train_4.reshape(len(y_train), 1)
    y_test_4 = y_test_4.reshape(len(y_test),1)

    
    y_train_total = np.vstack((y_train_1, y_train_2, y_train_3, y_train_4))
    y_test_total = np.vstack((y_test_1, y_test_2, y_test_3, y_test_4))
    X_train_total = np.vstack((X_train, X_train, X_train, X_train))
    X_test_total = np.vstack((X_test, X_test, X_test, X_test))
    
    sse_train_list = []
    sse_train_list_1, sse_train_list_2, sse_train_list_3, sse_train_list_4=[],[],[],[]
    sse_val_list=[]
    sse_val_list_1, sse_val_list_2, sse_val_list_3, sse_val_list_4=[],[],[],[]
    sse=1
    sse_val=1
    sse_prev=0.01
    sse_1=1
    sse_prev_1=0.01
    sse_2=1
    sse_prev_2=0.01
    sse_3=1
    sse_prev_3=0.01
    sse_4=1
    sse_prev_4=0.01
    xi_sh,xi_1,xi_2,xi_3, xi_4=0,0,0,0,0
    sse_val, sse_val_1, sse_val_2, sse_val_3, sse_val_4=1,1,1,1,1
    avg_mse=np.inf
    avg_mse_prev=0.01
    
    start_time = time.time()
    clf = dan2.DAN2Regressor()
    f = clf.add_layer(X=X_train_total,y=y_train_total,isFirstLayer=True,f_k=0)
    while ((avg_mse-avg_mse_prev)/avg_mse_prev)>TOLERANCE:
        print('sse is',(sse_val_1+sse_val_2+sse_val_3+sse_val_4)/4)
        avg_mse_prev = (sse_val_1+sse_val_2+sse_val_3+sse_val_4)/4

        while np.abs(sse_prev - sse)/sse_prev > TOLERANCE_1:
            sse_prev = sse
            f = clf.add_layer(X_train_total, y_train_total, False, f)
            sse = np.sum((f-y_train_total)**2)/y_train_total.shape[0]
            sse_train_list.append(sse)
            xi_sh = xi_sh+1
            f_val = clf.predict_shared(X_test_total)
            sse_val = np.sum((f_val-y_test_total)**2)/y_test_total.shape[0]
            sse_val_list.append(sse_val)
            print('Number of shared layers is', xi_sh)
            print('new sse is',sse)
            print('sseval is',sse_val)
        while np.abs(sse_1-sse_prev_1)/sse_prev_1>TOLERANCE_ts_1:
            if sse_val_1-sse_prev_1>100:
                break
            sse_prev_1=sse_1
            f_ts_1 = clf.add_layer_multihead(X_train, y_train_1,0, f[0:17094])
            sse_1 = np.sum((f_ts_1-y_train_1)**2)/y_train_1.shape[0]
            sse_train_list_1.append(sse_1)
            xi_1 = xi_1+1
            f_ts_val_1 = clf.predict_multihead(X_test,f_val[:7327],0)
            sse_val_1 = np.sum((f_ts_val_1-y_test_1)**2)/y_test_1.shape[0]
            sse_val_list_1.append(sse_val_1)
            print('Number of layers for Task 1 is', xi_1)
            print('sse1 is',sse_1)
            print('sseval is',sse_val_1)
        while np.abs(sse_2-sse_prev_2)/sse_prev_2>TOLERANCE_ts_2:
            if sse_val_2-sse_prev_2>100:
                break
            sse_prev_2=sse_2
            f_ts_2 = clf.add_layer_multihead(X_train, y_train_2,1, f[17094:17094*2])
            sse_2 = np.sum((f_ts_2-y_train_2)**2)/y_train_2.shape[0]
            sse_train_list_2.append(sse_2)
            xi_2 = xi_2+1
            f_ts_val_2 = clf.predict_multihead(X_test,f_val[7327:7327*2],1)
            sse_val_2 = np.sum((f_ts_val_2-y_test_2)**2)/y_test_2.shape[0]
            sse_val_list_2.append(sse_val_2)
            print('Number of layers for Task 2 is', xi_2)
            print('sse2 is',sse_2)
            print('sseval is',sse_val_2)
        while np.abs(sse_3-sse_prev_3)/sse_prev_3>TOLERANCE_ts_3:
            if sse_val_3-sse_prev_3>100:
                break
            sse_prev_3=sse_3
            f_ts_3 = clf.add_layer_multihead(X_train, y_train_3,2, f[17094*2:17094*3])
            sse_3 = np.sum((f_ts_3-y_train_3)**2)/y_train_3.shape[0]
            sse_train_list_3.append(sse_3)
            xi_3 = xi_3+1
            f_ts_val_3 = clf.predict_multihead(X_test,f_val[7327*2:7327*3],2)
            sse_val_3 = np.sum((f_ts_val_3-y_test_3)**2)/y_test_3.shape[0]
            sse_val_list_3.append(sse_val_3)
            print('Number of layers for Task 3 is', xi_3)
            print('sse3 is',sse_3)
            print('sseval is',sse_val_3)
        while np.abs(sse_4-sse_prev_4)/sse_prev_4>TOLERANCE_ts_4:
            if sse_val_4-sse_prev_4>100:
                break
            sse_prev_4=sse_4
            f_ts_4 = clf.add_layer_multihead(X_train, y_train_4,3, f[17094*3:])
            sse_4 = np.sum((f_ts_4-y_train_4)**2)/y_train_4.shape[0]
            sse_train_list_4.append(sse_4)
            xi_4 = xi_4+1
            f_ts_val_4 = clf.predict_multihead(X_test,f_val[7327*3:],3)
            sse_val_4 = np.sum((f_ts_val_4-y_test_4)**2)/y_test_4.shape[0]
            sse_val_list_4.append(sse_val_4)
            print('Number of layers for Task 3 is', xi_4)
            print('sse4 is',sse_4)
            print('sseval is',sse_val_4)
        avg_mse=(sse_val_1+sse_val_2+sse_val_3+sse_val_4)/4
    end_time =time.time()

    list_1 = sse_val_list+sse_val_list_1
    list_2 = sse_val_list+sse_val_list_2
    list_3 = sse_val_list+sse_val_list_3
    list_4 = sse_val_list+sse_val_list_4
    print('list 1',list_1)
    print('list 2',list_2)
    print('list 3',list_3)
    print('list 4',list_4)
  
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

    print('Shared Layers:',xi_sh)
    print('Task 1 Layer:', xi_1)
    print('Task 2 Layer', xi_2)
    print('Task 3 Layer', xi_3)
    print('Task 4 Layer', xi_4)
    print('Training Loss',sse, sse_1, sse_2, sse_3, sse_4)
    print('Validation Loss', sse_val, sse_val_1, sse_val_2, sse_val_3, sse_val_4)
    print('Average training SSE', sse_1+sse_2+sse_3+sse_4)
    print('Average validation MSE', (sse_val_1+sse_val_2+sse_val_3+sse_val_4)/3)
    print('Total time is:', end_time- start_time)


    plt.plot(list_1,label='Task 1',c='m')
    plt.plot(list_2,label='Task 2',c='r')
    plt.plot(list_3,label='Task 3',c='b')
    plt.plot(list_4,label='Task 4', c='k')
    plt.xlabel('Number of Iterations')
    plt.legend()
    plt.ylabel('The Validation Loss')
    plt.title('Architectural Optimisation using Validation Loss')
    plt.show()

    train_list_1 = sse_train_list+sse_train_list_1
    train_list_2 = sse_train_list+sse_train_list_2
    train_list_3 = sse_train_list+sse_train_list_3
    train_list_4 = sse_train_list+sse_train_list_4
    plt.plot(train_list_1,label='Task 1',c='m')
    plt.plot(train_list_2,label='Task 2',c='r')
    plt.plot(train_list_3,label='Task 3',c='b')
    plt.plot(train_list_4,label='Task 4', c='k')
    plt.xlabel('Number of Iterations')
    plt.legend()
    plt.ylabel('The Training Loss')
    plt.title('Training Loss vs Number of Iterations')
    plt.show()

    # mse_list.append((mse_1,mse_2,mse_3))
    # tst_mse_1, perc_error_1 = test(X_test_total, y_test_total, 0, clf)
    # tst_mse_2, perc_error_2 = test(X_test_total, y_test_total, 1, clf)
    # tst_mse_3, perc_error_3 = test(X_test_total, y_test_total, 2,clf)
    # print("training mse", mse_1, mse_2, mse_3)
    # print("MSE",  tst_mse_1,tst_mse_2,tst_mse_3)
    # print("APE ", perc_error_1, perc_error_2, perc_error_3)
