"""
Finding the best number of layers with validation set
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
    tr_pred, mse_train = clf.fit(X_train, y_train)
    path = clf.name
    save_file(path, clf)
    print(clf.coef_)
    y_val_pred = clf.predict(X_val)
    mse_val = np.sum((y_val_pred - y_val)**2) / X_val.shape[0]   
    # print('True prediction is',y_test, 'Model predicts', y_pred)
    # print('Difference',(y_pred-y_test)/y_pred)
    # print(test_fit_and_predict(y_test, y_pred))

    print('sum percentage diff', sum(np.abs(y_val_pred-y_val)/y_val)/100)
    return mse_train, mse_val




if __name__ == '__main__':
    # X = load_file(sys.argv[1])
    # y = load_file(sys.argv[2])
    X = load_file(r'/Users/sushenzhang/Documents/phd/second_year_code/X_values.pkl')
    y = load_file(r'/Users/sushenzhang/Documents/phd/second_year_code/Y_values.pkl')
    X = (X-X.min())/(X.max()-X.min())
    y= (y-y.min())/(y.max()-y.min())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.30, random_state=42)
    y_train_0 = y_train[:,3]
    y_val_0 = y_val[:,3]
    y_test_0 = y_test[:,3]
    y_train_0 = y_train_0.reshape(len(y_train_0), 1)
    y_val_0 = y_val_0.reshape(len(y_val_0), 1)
    y_test_0 = y_test_0.reshape(len(y_test_0),1)
    

    mse_val_list =[0] * 50
    mse_train_list=[0] *50
    for j in range(30):
	    for i in range(1,50):
	    	mse_train, mse_val=main(X_train, X_val, y_train_0, y_val_0, i)
	    	mse_val_list[i]+=mse_val
	    	mse_train_list[i]+=mse_train
    mse_val_list= [item/30 for item in mse_val_list]
    mse_train_list= [item/30 for item in mse_train_list]
    mse_val_list.pop(0)
    mse_train_list.pop(0)
	

    plt.plot(mse_val_list)
    # plt.plot(mse_val_list_1,'g',label='Purity')
    # plt.plot(mse_val_list_3,'b',label = 'Energy Consumption')
    plt.legend()
    plt.title('The MSE of Validation Set over the Number of Hidden Layers')
    plt.xlabel('Number of Hidden Layers')
    plt.ylabel('MSE of Validation Set')
    plt.show()
    plt.plot(mse_train_list)
    # plt.plot(mse_train_list_1,'g',label='Purity')
    # plt.plot(mse_train_list_3,'b',label = 'Energy Consumption')
    plt.legend()
    plt.title('The MSE of Training Set over the Number of Hidden Layers')
    plt.xlabel('Number of Hidden Layers')
    plt.ylabel('MSE of Training Set')
    plt.show()
    print(y_test)
"""		
    y_train_1 = y_train[:,1]
    y_val_1 = y_val[:,1]
    y_test_1 =y_test[:,1]
    y_train_1 = y_train_1.reshape(len(y_train_1), 1)
    y_val_1 = y_val_1.reshape(len(y_val_1), 1)
    y_test_1 = y_test_1.reshape(len(y_test_1),1)
    mse_val_list_1 =[]
    mse_train_list_1 = []
    for i in range(1,50):
    	mse_train, mse_val=main(X_train, X_val, y_train_1,y_val_1,i)
    	mse_val_list_1.append(mse_val)
    	mse_train_list_1.append(mse_train)
    
		
    y_train_3 = y_train[:,3]
    y_val_3 = y_val[:,3]
    y_test_3 = y_test[:,3]
    y_train_3 = y_train_3.reshape(len(y_train_3), 1)
    y_val_3 = y_val_3.reshape(len(y_val_3), 1)
    y_test_3 = y_test_3.reshape(len(y_test_3),1)
    mse_val_list_3 =[]
    mse_train_list_3 = []
    for i in range(1,50):
    	mse_train, mse_val=main(X_train, X_val, y_train_3,y_val_3,i)
    	mse_val_list_3.append(mse_val)
    	mse_train_list_3.append(mse_train)

"""



'''		
print('model_weights', clf.model['weights'])
print('model_intercepts', clf.model['intercept'])
print('model_mu', clf.model['mu'])
'''