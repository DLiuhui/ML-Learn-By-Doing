import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def training(x, y, w, iteration=1000, alpha=1e-3, eps=1e-10):
    total = len(y)
    # gradient descent
    rmse = []
    for i in range(iteration):
        y_pre = np.dot(x,w)
        loss = y_pre - y
        w_new = w - alpha / total * np.dot(x.T, loss)
        rmse.append((np.sqrt(np.dot(loss.T, loss) / total)).item())
        if(np.max(np.abs(w_new - w)) < eps):
            break
        w = w_new
    return rmse, w

if __name__=="__main__":
    train_path = "./data/train.csv"
    test_path = "./data/test.csv"
    train_raw = pd.read_csv(train_path, sep=',', encoding='big5')
    test_data = pd.read_csv(test_path, sep=',', encoding='big5')
    train_1 = train_raw.iloc[:, 3:3+10]
    train_2 = train_raw.iloc[:, 16:16+10]
    attr_index = train_raw.iloc[:, 2]
    train_1 = pd.concat([train_1, attr_index], axis=1)
    train_2 = pd.concat([train_2, attr_index], axis=1)
    y_1 = train_1[train_1["測項"] == "PM2.5"]
    y_1 = y_1['9'].values
    y_2 = train_2[train_2["測項"] == "PM2.5"]
    y_2 = y_2['22'].values
    Y = np.concatenate((y_1, y_2), axis=0)
    Y = Y.astype('float64')
    Y = Y[:,np.newaxis]
    del y_1, y_2
    x_1 = train_1.iloc[:,0:9]
    x_2 = train_2.iloc[:,0:9]
    x_1.replace('NR',0,inplace=True)
    x_2.replace('NR',0,inplace=True)
    x_1 = x_1.values.reshape((240, -1))
    x_2 = x_2.values.reshape((240, -1))
    X_raw = np.concatenate((x_1, x_2), axis=0)
    X_raw = X_raw.astype('float64')
    tmp = np.ones(Y.shape)
    X = np.c_[tmp, X_raw]
    del tmp
    # w = np.random.randint(50, 75, size=(X.shape[1], 1))
    w = np.zeros((X.shape[1], 1))
    rmse, w_end = training(X, Y, w, iteration=2000, alpha=3e-6)
    print(rmse)
    plt.figure(0)
    plt.plot(range(len(rmse)), rmse)
    plt.show()
