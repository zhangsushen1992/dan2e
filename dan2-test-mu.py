"""
This serves to compare fixed mu and random mu
"""

import dan2_fixed_mu as dan2_f
import dan2 as dan2
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


def save_file(path, model):
    with open(path + '.pk', 'wb') as save_file:
        pickle.dump(model, save_file)

def test_fit_and_predict(training_preds, testing_preds):
    return np.array_equal(training_preds, testing_preds)

def main(X, X_test, y_train, y_test, depth):
    clf = dan2.DAN2Regressor(depth=depth)
    tr_pred = clf.fit(X_train, y_train)
    path = clf.name
    save_file(path, clf)
    print(clf.coef_)
    y_pred = clf.predict(X_test)
    mse = np.sum((y_pred - y_test)**2) / X_test.shape[0]
    perc_error=sum(np.abs((y_pred-y_test)/y_test))/(3264*0.3)
    return perc_error, mse,y_pred

def main_2(X, X_test, y_train, y_test, depth):
    clf = dan2_f.DAN2Regressor(depth=depth)
    tr_pred = clf.fit(X_train, y_train)
    path = clf.name
    save_file(path, clf)
    print(clf.coef_)
    y_pred = clf.predict(X_test)
    mse = np.sum((y_pred - y_test)**2) / X_test.shape[0]
    perc_error=sum(np.abs((y_pred-y_test)/y_test))/(3264*0.3)
    return perc_error, mse,y_pred



if __name__ == '__main__':
    # X = load_file(sys.argv[1])
    # y = load_file(sys.argv[2])
    X = load_file(r'/Users/sushenzhang/Documents/phd/second_year_code/X_values.pkl')
    y = load_file(r'/Users/sushenzhang/Documents/phd/second_year_code/Y_values.pkl')
    X = (X-X.min())/(X.max()-X.min())
    y= (y-y.min())/(y.max()-y.min())
    # X = (X-np.mean(X,axis=0))/np.std(X,axis=0)
    # y = (y-np.mean(y,axis=0))/np.std(y,axis=0)
   
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.30, random_state=42)
   
    y_train = y_train[:,1]
    y_test = y_test[:,1]
    y_train = y_train.reshape(len(y_train), 1)
    y_test = y_test.reshape(len(y_test),1)
    
    mse_list =[]

    perc_error, mse, y_pred=main_2(X_train, X_test, y_train,y_test,50)
    perc_error_f, mse_f, y_pred_f = main(X_train, X_test, y_train,y_test,50)
    perc_error_f_1, mse_f_1, y_pred_f_1 = main(X_train, X_test, y_train,y_test,50)
    perc_error_f_2, mse_f_2, y_pred_f_2 = main(X_train, X_test, y_train,y_test,50)
    mse_list.append(mse)


    plt.plot(y_pred-y_test,lw=0.5,c='b',label='Predicted Values with Fixed μ')
    plt.plot(y_pred_f-y_test,lw=0.5,c='g',label='Predicted Values with Random μ (Set 1)')
    plt.plot(y_pred_f_1-y_test,lw=0.5,c='r',label='Predicted Values with Random μ (Set 2)')
    plt.plot(y_pred_f_2-y_test,lw=0.5,c='m',label='Predicted Values with Random μ(Set 3)')
    plt.legend()
    plt.title('The Effect of Fixed and Random μ')
    plt.xlabel('Testing Dataset')
    plt.ylabel('Recovery Rate (%)')
    plt.show()
    print('avg error is:',np.sum(np.abs((y_pred-y_test)/y_test))/3264*0.3)
    print ('error',mse, perc_error)
    print('fixed error', mse_f, perc_error_f)
    print('fixed error 1', mse_f_1, perc_error_f_1)
    print('fixed error 2', mse_f_2,perc_error_f_2)
    # print(X.min(),X.max())
    print(y_test.max(),y_test.min())
    # print(np.mean(X,axis=0),np.std(X,axis=0,dtype = np.float32))
    # print(np.mean(y,axis=0),np.std(y,axis=0,dtype = np.float32))
'''		
print('model_weights', clf.model['weights'])
print('model_intercepts', clf.model['intercept'])
print('model_mu', clf.model['mu'])
'''