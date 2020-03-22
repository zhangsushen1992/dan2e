'''
Testing the effect of dropping out in the network
'''
import dan2_dropout as dan2
import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from dan2_dropout import dropout



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
    clf.dropout(0.95)
    # path = clf.name
    # save_file(path, clf)
    # print(clf.coef_)
    y_pred = clf.predict(X_test)
    mse = np.sum((y_pred - y_test)**2) / X_test.shape[0]

    perc_error=sum(np.abs((y_pred-y_test)/y_test))/(3264*0.3)
    return perc_error, mse,y_pred




if __name__ == '__main__':
  
    X = load_file(r'/Users/sushenzhang/Documents/phd/second_year_code/X_values.pkl')
    y = load_file(r'/Users/sushenzhang/Documents/phd/second_year_code/Y_values.pkl')
    X = (X-X.min())/(X.max()-X.min())
    y= (y-y.min())/(y.max()-y.min())
   

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.30, random_state=42)

    y_train = y_train[:,0]
    y_test = y_test[:,0]
    y_train = y_train.reshape(len(y_train), 1)
    y_test = y_test.reshape(len(y_test),1)

    mse_list =[]

    perc_error, mse, y_pred=main(X_train, X_test, y_train,y_test,19)
    mse_list.append(mse)


    plt.plot(y_pred,lw=0.5,c='b',label='Predicted Values of the Recovery Rate')
    plt.plot(y_test,lw=0.5,c='g',label='True Values of the Recovery Rate')
    plt.legend()
    plt.xlabel('Testing Dataset')
    plt.ylabel('Recovery Rate (%)')
    plt.show()
    print('avg error is:',np.sum(np.abs((y_pred-y_test)/y_test))/3264*0.3)
    print(y_test)
    print ('error',mse, perc_error)

    print(y_test.max(),y_test.min())