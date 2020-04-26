"""
This file serves to optimise a structure with defined number of hidden layers
"""


# import dan2_original as dan2
import dan2 as dan2
import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
'''
# Test dataset for classification
with open('data/x-train.P', 'rb') as f:
    X_train = pickle.load(f)
    X_train = np.array(X_train)

with open('data/y-train.P', 'rb') as f:
    y_train = pickle.load(f)
    y_train = np.array(y_train)
    #y_train = np.where(y_train==1, 100, -100)

print(X_train)
y_train = y_train.reshape(len(y_train), 1)
print(y_train)

clf = dan2.DAN2Regressor(depth=10)
clf.fit(X_train, y_train, f_0 = None)
'''



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
    # path = clf.name
    # save_file(path, clf)
    # print(clf.coef_)
    y_pred = clf.predict(X_test)
    mse = np.sum((y_pred - y_test)**2) / X_test.shape[0]
    # print('True prediction is',y_test, 'Model predicts', y_pred)
    # print('Difference',(y_pred-y_test)/y_pred)
    # print(test_fit_and_predict(y_test, y_pred))
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
   
    # plt.hist(X)
    # plt.show()
    # plt.hist(y)
    # plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.30, random_state=42)
    # X_train = X[:-100]
    # X_test = X[-100:]
    # y_train = y[:-100,3]
    # y_test = y[-100:,3]
    y_train = y_train[:,3]
    y_test = y_test[:,3]
    y_train = y_train.reshape(len(y_train), 1)
    y_test = y_test.reshape(len(y_test),1)
    # main(X, y, int(sys.argv[3]))
    mse_list =[]

    perc_error, mse, y_pred=main(X_train, X_test, y_train,y_test,33)
    mse_list.append(mse)


    plt.plot(y_pred,lw=0.5,c='b',label='Predicted Values of the Recovery Rate')
    plt.plot(y_test,lw=0.5,c='g',label='True Values of the Recover Rate')
    plt.legend()
    plt.xlabel('Testing Dataset')
    plt.ylabel('Recovery Rate (%)')
    plt.show()
    print('avg error is:',np.sum(np.abs((y_pred-y_test)/y_test))/3264*0.3)
    print(y_test)
    print ('error',mse, perc_error)
    # print(X.min(),X.max())
    print(y_test.max(),y_test.min())
    # print(np.mean(X,axis=0),np.std(X,axis=0,dtype = np.float32))
    # print(np.mean(y,axis=0),np.std(y,axis=0,dtype = np.float32))
'''		
print('model_weights', clf.model['weights'])
print('model_intercepts', clf.model['intercept'])
print('model_mu', clf.model['mu'])
'''