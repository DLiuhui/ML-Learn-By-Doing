import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, recall_score, f1_score, precision_score

def dataPreProRegression(train_path, test_path, sep=",",  IQR_limit = 1.5, corr_line = 0.075, skew_line = 1.0):
    # reading file
    try:
        train_raw_data = pd.read_csv(train_path, sep=sep)
    except IOError:
        print("%s invalid"%train_path)
        exit(1)
    try:
        test_raw_data = pd.read_csv(test_path, sep=sep)
    except IOError:
        print("%s invalid"%test_path)
        exit(1)
    attrCon = ["ID", "Reason for absence", "Transportation expense", "Distance from Residence to Work", "Service time", "Age", "Work load Average/day ", "Hit target", "Son", "Pet", "Weight", "Height", "Body mass index", "Absenteeism time in hours"]
    attrCat = ["Month of absence", "Day of the week", "Seasons", "Education"]
    attrBool = ["Disciplinary failure", "Social drinker", "Social smoker"]
    # begin
    train_raw_data.drop(columns=["ID", "Weight", "Seasons", "Service time"], inplace=True)
    test_raw_data.drop(columns=["ID", "Weight", "Seasons", "Service time"], inplace=True)
    # deal with skew
    featureCon = ["Reason for absence", "Transportation expense", "Distance from Residence to Work", "Age", "Work load Average/day ", "Hit target", "Son", "Pet", "Height", "Body mass index"]
    for s in featureCon:
        if(abs(train_raw_data[s].skew()) > skew_line):
            train_raw_data[s] = train_raw_data[s].apply(lambda x : np.log(x + 1))
            test_raw_data[s] = test_raw_data[s].apply(lambda x : np.log(x + 1))
    # delete abnormal data
    featureAbn = featureCon
    # featureAbn = ["Absenteeism time in hours"]
    index = set(train_raw_data.index)
    for s in featureAbn:
        percentile = np.percentile(train_raw_data[s], [0, 25, 50, 75, 100])
        IQR = percentile[3] - percentile[1]
        upLimit = percentile[3]+IQR*IQR_limit
        downLimit = percentile[1]-IQR*IQR_limit
        for idx in index:
            if(train_raw_data.loc[idx][s] > upLimit or train_raw_data.loc[idx][s] < downLimit):
                train_raw_data.drop([idx], inplace=True)
                index = index - set([idx])
    # Correlation
    corr_matrix = train_raw_data.corr()
    # decide attr to use
    attrs = set(corr_matrix.columns) - set(["Absenteeism time in hours", "Service time"])
    using_attrs = []
    for attr in attrs:
        if(abs(corr_matrix["Absenteeism time in hours"][attr]) > corr_line):
            using_attrs.append(attr)
    # data solve
    n_train = len(train_raw_data)
    n_test = len(test_raw_data)

    raw_data = pd.concat([train_raw_data, test_raw_data])

    using_con = []
    using_cat = []
    using_bool = []
    for attr in using_attrs:
        if(attr in attrCon):
            using_con.append(attr)
        elif(attr in attrCat):
            using_cat.append(attr)
        else:
            using_bool.append(attr)

    X_vec_con = raw_data[using_con].values
    X_vec_cat = raw_data[using_cat].values
    X_vec_bool = raw_data[using_bool].values

    if(X_vec_con.shape[1] > 0):
        scaler=preprocessing.StandardScaler().fit(X_vec_con)
        # scaler=preprocessing.MinMaxScaler().fit(X_vec_con)
        X_vec_con_ed=scaler.transform(X_vec_con)
    else:
        X_vec_con_ed = X_vec_con
        
    if(X_vec_cat.shape[1] > 0):
        enc=preprocessing.OneHotEncoder(categories='auto')
        enc.fit(X_vec_cat)
        X_vec_cat_ed=enc.transform(X_vec_cat).toarray()
    else:
        X_vec_cat_ed = X_vec_cat
        
    X_vec = np.concatenate((X_vec_con_ed, X_vec_cat_ed, X_vec_bool), axis=1)
    Y_vec = raw_data["Absenteeism time in hours"].values[:, np.newaxis]

    # split train and test
    x_train = X_vec[0 : n_train]
    y_train = Y_vec[0 : n_train]
    x_test = X_vec[n_train : n_test+n_train]
    y_test = Y_vec[n_train : n_test+n_train]

    return x_train, y_train, x_test, y_test

def linearRegression(x_train, y_train, x_test, y_test):
    linreg = LinearRegression().fit(x_train, y_train)
    y_pre = linreg.predict(x_test)
    mse = mean_squared_error(y_test, y_pre)
    rmse = np.sqrt(mse)
    r2_train = linreg.score(x_train, y_train)
    r2_test = linreg.score(x_test, y_test)
    # print('MSE = ', mse)
    # print('RMSE = ', rmse)
    # print('R2 = ', r2_train)
    # print('R2 = ', r2_test)
    return mse, rmse, r2_train, r2_test
    # plt.figure(2)
    # t = np.arange(len(x_test))
    # plt.plot(t, y_test, 'r-', linewidth=2, label="real data")
    # plt.plot(t, y_pre, 'g-', linewidth=2, label="predict data")
    # plt.legend(loc='upper right')
    # plt.title("linear regression", fontsize=10)
    # plt.grid(b=True)
    # plt.show()

def svmRegression(x_train, y_train, x_test, y_test):
    # using svr
    clf = SVR(kernel="linear", C=1.0, epsilon=0.1)
    clf.fit(x_train, y_train)
    y_pre = clf.predict(x_test)
    mse = mean_squared_error(y_test, y_pre)
    rmse = np.sqrt(mse)
    r2_train = clf.score(x_train, y_train)
    r2_test = clf.score(x_test, y_test)
    # print('MSE = ', mse)
    # print('RMSE = ', rmse)
    # print('R2 = ', r2_train)
    # print('R2 = ', r2_test)
    return mse, rmse, r2_train, r2_test
    # plt.figure(3)
    # t = np.arange(len(x_test))
    # plt.plot(t, y_test, 'r-', linewidth=2, label="real data")
    # plt.plot(t, y_pre, 'g-', linewidth=2, label="predict data")
    # plt.legend(loc='upper right')
    # plt.title("linear regression", fontsize=10)
    # plt.grid(b=True)
    # plt.show()

def dataPreProClassification(train_path, test_path, sep=",",  IQR_limit = 1.5, corr_line = 0.075, skew_line = 1.0):
    # reading file
    try:
        train_raw_data = pd.read_csv(train_path, sep=sep)
    except IOError:
        print("%s invalid"%train_path)
        exit(1)
    try:
        test_raw_data = pd.read_csv(test_path, sep=sep)
    except IOError:
        print("%s invalid"%test_path)
        exit(1)
    attrCon = ["ID", "Reason for absence", "Transportation expense", "Distance from Residence to Work", "Service time", "Age", "Work load Average/day ", "Hit target", "Son", "Pet", "Weight", "Height", "Body mass index", "Absenteeism time in hours"]
    attrCat = ["Month of absence", "Day of the week", "Seasons", "Education"]
    attrBool = ["Disciplinary failure", "Social drinker", "Social smoker"]
    # begin
    train_raw_data.drop(columns=["Weight", "Service time"], inplace=True)
    test_raw_data.drop(columns=["Weight", "Service time"], inplace=True)
    featureCon = ["Absenteeism time in hours", "Transportation expense", "Distance from Residence to Work", "Age", "Work load Average/day ", "Hit target", "Son", "Pet", "Height", "Body mass index"]
    # deal with skew
    for s in featureCon:
        if(abs(train_raw_data[s].skew()) > skew_line):
            train_raw_data[s] = train_raw_data[s].apply(lambda x : np.log(x + 1))
            test_raw_data[s] = test_raw_data[s].apply(lambda x : np.log(x + 1))
    # delete abnormal data
    # featureAbn = list(set(featureCon) - set(["Absenteeism time in hours"]))
    featureAbn = featureCon
    index = set(train_raw_data.index)
    for s in featureAbn:
        percentile = np.percentile(train_raw_data[s], [0, 25, 50, 75, 100])
        IQR = percentile[3] - percentile[1]
        upLimit = percentile[3]+IQR*IQR_limit
        downLimit = percentile[1]-IQR*IQR_limit
        for idx in index:
            if(train_raw_data.loc[idx][s] > upLimit or train_raw_data.loc[idx][s] < downLimit):
                train_raw_data.drop([idx], inplace=True)
                index = index - set([idx])
    corr_matrix = train_raw_data.corr()
    # decide attr to use
    attrs = set(corr_matrix.columns) - set(["Reason for absence"])
    using_attrs = []
    for attr in attrs:
        if(abs(corr_matrix["Reason for absence"][attr]) > corr_line):
            using_attrs.append(attr)
    # preprocessing
    n_train = len(train_raw_data)
    n_test = len(test_raw_data)

    raw_data = pd.concat([train_raw_data, test_raw_data])

    using_con = []
    using_cat = []
    using_bool = []
    for attr in using_attrs:
        if(attr in attrCon):
            using_con.append(attr)
        elif(attr in attrCat):
            using_cat.append(attr)
        else:
            using_bool.append(attr)

    X_vec_con = raw_data[using_con].values
    X_vec_cat = raw_data[using_cat].values
    X_vec_bool = raw_data[using_bool].values

    if(X_vec_con.shape[1] > 0):
        scaler=preprocessing.StandardScaler().fit(X_vec_con)
        X_vec_con_ed=scaler.transform(X_vec_con)
    else:
        X_vec_con_ed = X_vec_con
        
    if(X_vec_cat.shape[1] > 0):
        enc=preprocessing.OneHotEncoder(categories='auto')
        enc.fit(X_vec_cat)
        X_vec_cat_ed=enc.transform(X_vec_cat).toarray()
    else:
        X_vec_cat_ed = X_vec_cat
        
    X_vec = np.concatenate((X_vec_con_ed, X_vec_cat_ed, X_vec_bool), axis=1)
    Y_vec = raw_data["Reason for absence"].values[:, np.newaxis]
    #split
    x_train = X_vec[0 : n_train]
    y_train = Y_vec[0 : n_train]
    x_test = X_vec[n_train : n_test+n_train]
    y_test = Y_vec[n_train : n_test+n_train]
    return x_train, y_train, x_test, y_test

def svmClassification(x_train, y_train, x_test, y_test):
    clf = SVC(kernel="linear")
    clf.fit(x_train, y_train)
    y_pre = clf.predict(x_test)
    recall = recall_score(y_test, y_pre, average='micro')
    acc = precision_score(y_test, y_pre, average='micro') 
    f1 = f1_score(y_test, y_pre, average='micro')
    # print("Accuracy = ", acc)
    # print('Recall = ', recall)
    # print('F1 = ', f1)
    return acc, recall, f1