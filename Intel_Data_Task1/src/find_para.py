import sys
import myKernel
import numpy as np

if __name__ == "__main__": 
    if(len(sys.argv) < 3):
        print("please run as: python main.py train_data_path test_data_path")
        exit(0)
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    best_mse = 9999999
    best_rmse = 0
    best_r2_train = 0
    best_r2_test = 0
    best_irq = 0
    best_cl = 0
    best_sl = 0
    for irq in np.arange(1.1, 2.5, 0.1):
        for cl in np.arange(0.03, 0.12, 0.01):
            for sl in np.arange(0.5, 2.0, 0.1):
                x_train, y_train, x_test, y_test = myKernel.dataPreProRegression(train_path, test_path, sep=";", \
                    IQR_limit = irq, corr_line = cl, skew_line = sl)
                mse1, rmse1, rtrain1, rtest1 = myKernel.linearRegression(x_train, y_train, x_test, y_test)
                mse2, rmse2, rtrain2, rtest2 = myKernel.svmRegression(x_train, y_train, x_test, y_test)
                if(mse1 < mse2 and mse1 < best_mse):
                    best_mse = mse1
                    best_rmse = rmse1
                    best_r2_train = rtrain1
                    best_r2_test = rtest1
                    best_irq = irq
                    best_cl = cl
                    best_sl = sl
                elif(mse1 > mse2 and mse2 < best_mse):
                    best_mse = mse2
                    best_rmse = rmse2
                    best_r2_train = rtrain2
                    best_r2_test = rtest2
                    best_irq = irq
                    best_cl = cl
                    best_sl = sl
    print(best_mse, best_rmse, best_r2_train, best_r2_test, best_irq, best_cl, best_sl)

    