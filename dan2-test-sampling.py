"""
This file serves to optimise a structure with stochastic sampling
"""

import dan2_sampling as dan2
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

def main(X_train, X_val, y_train, y_val, depth):
    clf = dan2.DAN2Regressor(depth=depth)
    tr_pred = clf.fit(X_train, y_train)
    # path = clf.name
    # save_file(path, clf)
    print(clf.coef_)
    y_val_pred = clf.predict(X_val)
    discrepency = np.sum((y_val_pred - y_val)**2)/y_val_pred.shape[0]
    print('discrepency is', discrepency)
    sigma = np.std((y_val_pred-y_val)**2)
    print ('sigma is ',sigma)
    # weightings = (sigma)**(-BATCH_SIZE)*np.exp(-sigma**(-2)*discrepency/2)
    weightings = np.exp(discrepency)
    # print('blablabla',np.exp(-sigma**(-2)*discrepency/2) )
    mse =  discrepency# / X_test.shape[0]
    perc_error=sum(np.abs((y_val_pred-y_val)/y_val))/y_val.shape[0]
    return perc_error, mse,y_val_pred, clf.coef_, weightings,clf


def testing(model,X_test, y_test, coef, depth):
    
    model.read_coefficients(coef)
    y_test_pred=model.predict(X_test)
    mse=np.sum((y_test_pred - y_test)**2)/y_test.shape[0]
    perc_error=sum(np.abs((y_test_pred-y_test)/y_test))/y_test.shape[0]
    return y_test_pred, mse, perc_error

if __name__ == '__main__':
 
    X = load_file(r'/Users/sushenzhang/Documents/phd/second_year_code/X_values.pkl')
    y = load_file(r'/Users/sushenzhang/Documents/phd/second_year_code/Y_values.pkl')
    X = (X-X.min())/(X.max()-X.min())
    y= (y-y.min())/(y.max()-y.min())


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33333, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    BATCH_SIZE = 197*2
    mse_list =[]
    weightings_list = []
    coefficients_list = []
    coef_sum = 0

    for i in range (0,3):
        X_train_batch = X_train[(BATCH_SIZE*i):(BATCH_SIZE*(i+1))]
        X_val_batch = X_val#[(BATCH_SIZE*i):(BATCH_SIZE*(i+1))]
        y_train_batch = y_train[(BATCH_SIZE*i):(BATCH_SIZE*(i+1)),0]
        y_val_batch = y_val[:,0]#[(BATCH_SIZE*i):(BATCH_SIZE*(i+1)),0]
        y_train_batch = y_train_batch.reshape(len(y_train_batch), 1)
        y_val_batch = y_val_batch.reshape(len(y_val_batch),1)
  
        perc_error, mse, y_pred, coefficients, weightings,clf = main(X_train_batch, X_val_batch, y_train_batch, y_val_batch,19)
        coefficients_list.append(coefficients)
        weightings_list.append(weightings)
        mse_list.append(mse)

    # print(len(coefficients_list), len(coefficients_list[0]),len(coefficients_list[0][0]))
    # exit()
    
    print ('list of coefficients',coefficients_list)
    for j in range(0,3):
        # plt.plot(coefficients_list[j][:][0],label=("Coefficient μ of Batch "+str(j)))
        coef =coefficients_list[j]
        compare_list=coef[:,4].reshape(len(coefficients_list[j]),1)
        plt.plot(compare_list,label=("Coefficient c of Batch "+str(j)))
    # plt.plot(coefficients_list[:][:][1],label="Coefficent for CAKE node")
    plt.xlabel("The Number of Hidden Layer")
    plt.ylabel("The values of the Coefficients")
    plt.title("The Values of Coefficients in Each Batch")
    plt.legend()
    plt.show()
    '''
    y_test_batch = y_test[:,0]
    y_test_batch = y_test_batch.reshape(len(y_test_batch),1) 
    for coef_index in range(0,3):
        final_prediction,test_mse,test_perc_error=testing(clf, X_test, y_test_batch,coefficients_list[coef_index],19)
        print('test mse', test_mse,'test perc error', test_perc_error)
    print('training mse',mse_list)
    '''
    exit()

    sum_weight=np.sum(weightings_list)    
    for coef, weight in zip(coefficients_list, weightings_list):
        print('coef',coef)
        print ('weight',weight)
        coef_sum += coef / 6
        # coef_sum += coef*weight/sum_weight

    
    # max_index=np.argmax(weightings_list)
    # coef_sum=coefficients_list[max_index]   
    final_prediction,test_mse,test_perc_error=testing(clf, X_test, y_test_batch,coef_sum,19)
    print('weighted sum', coef_sum)
    print('ytest',y_test_batch)
    print('ceof')

    
    plt.plot(final_prediction,lw=0.5,c='b',label='Predicted Values of the Recovery Rate')
    plt.plot(y_test_batch,lw=0.5,c='g',label='True Values of the Recovery Rate')
    plt.legend()
    plt.xlabel('Testing Dataset')
    plt.ylabel('Recovery Rate (%)')
    plt.show()


    
    print('avg error is:',np.sum(np.abs((final_prediction-y_test[0])/y_test[0]))/y_test.shape[0])
    print(y_test[0])
    print ('error',mse_list, perc_error)
    print(y_test.max(),y_test.min())
