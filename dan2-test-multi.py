"""
Multitask testing
"""

import dan2_multi as dan2
import numpy as np
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
    
    mse_list = []
    # tst_mse_list_1 = []

    f_k, mse, clf = main(X_train_total, X_test_total, y_train_total ,y_test_total,20)
    # print ('-'*50, X_train, '-'*50, y_train_1, '-'*50, f_k, clf)
    mse_1, fk_1 = multi(X_train, y_train_1, 10, 0, f_k, clf)
    mse_2, fk_2 = multi(X_train, y_train_2, 20, 1, f_k, clf)
    mse_3, fk_3 = multi(X_train, y_train_3, 15, 2, f_k, clf)
    mse_list.append((mse_1,mse_2,mse_3))
    tst_mse_1, perc_error_1 = test(X_test_total, y_test_total, 0, clf)
    tst_mse_2, perc_error_2 = test(X_test_total, y_test_total, 1, clf)
    tst_mse_3, perc_error_3 = test(X_test_total, y_test_total, 2,clf)
    print("training mse", mse_1, mse_2, mse_3)
    print("MSE",  tst_mse_1,tst_mse_2,tst_mse_3)
    print("APE ", perc_error_1, perc_error_2, perc_error_3)
    """
    plt.plot(y_pred,lw=0.5,c='b',label='Predicted Values of the Recovery Rate')
    plt.plot(y_test,lw=0.5,c='g',label='True Values of the Recover Rate')
    plt.legend()
    plt.xlabel('Testing Dataset')
    plt.ylabel('Recovery Rate (%)')
    plt.show()
    print('avg error is:',np.sum(np.abs((y_pred-y_test)/y_test))/3264*0.3)
    print(y_test)
    print ('error',mse, perc_error)
  
    print(y_test.max(),y_test.min())
    """