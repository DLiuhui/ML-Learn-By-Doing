import sys
import myKernel
import numpy as np

if __name__ == "__main__": 
    if(len(sys.argv) < 3):
        print("please run as: python main.py train_data_path test_data_path")
        exit(0)
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    x_train, y_train, x_test, y_test = myKernel.dataPreProRegression(train_path, test_path, sep=";", \
        IQR_limit = 2.0, corr_line = 0.075, skew_line = 1.0)
    mse, rmse, r2_train, r2_test = myKernel.linearRegression(x_train, y_train, x_test, y_test)

    x_train, y_train, x_test, y_test = myKernel.dataPreProClassification(train_path, test_path, sep=";", \
        IQR_limit = 2.0, corr_line = 0.10, skew_line = 1.0)
    acc, recall, f1 = myKernel.svmClassification(x_train, y_train, x_test, y_test)

    print("Micro-average F1 of classification:")
    print("%.2f%%"%(f1*100))
    print("Mean squared error of regression:")
    print("%.2f"%(mse))

    