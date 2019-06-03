import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from auc_kernel import onlineAucMaximum
from sklearn.metrics import roc_curve, auc
from scipy import interp
import multiprocessing
import sys

N_BUFF_POS = 200
N_BUFF_NEG = 200
C = 15

def training(fold, x_data, y_data, queue, lock):
#     print("task", fold)
    mean_fpr = np.linspace(0, 1, 100)
    tmp = list(range(fold)) + list(range(fold+1,10))
    x_test = x_data[fold]
    y_test = y_data[fold]
    x_train = x_data[tmp[0]]
    y_train = y_data[tmp[0]]
    for idx in range(1, 9):
        x_train = np.concatenate((x_train, x_data[idx]))
        y_train = np.concatenate((y_train, y_data[idx]))
    w = onlineAucMaximum(x_train, y_train, N_BUFF_POS, N_BUFF_NEG, C)
    # predict
    y_pre = np.dot(x_test, w)
    # for auc
    fpr, tpr, thresholds = roc_curve(y_test, y_pre)
    roc_auc = auc(fpr, tpr)
    tprs_item = interp(mean_fpr, fpr, tpr)
    lock.acquire() # lock
    queue.put((w, tprs_item, roc_auc))
    lock.release() # unlock
        
def main():
    # get argv
    if(len(sys.argv) < 2):
        print("use: python3 exp.py train_path [test_path]")
        return 1
    elif(len(sys.argv) == 2):
        train_path = sys.argv[1]
    else:
        train_path = sys.argv[1]
        test_path = sys.argv[2]
    # load training data
    raw_data = loadmat(train_path)
    raw_x = raw_data['x_tr']
    raw_y = raw_data['y_tr']
    # std
    scaler = MinMaxScaler().fit(raw_x)
    scaler.transform(raw_x)
    # add bias
    bia = np.ones((len(raw_y), 1))
    raw_x = np.hstack((raw_x, bia))
    # split data set in 10 parts
    x_data = []
    y_data = []
    x_, y_ = raw_x, raw_y
    for fold in range(9, 0, -1):
        x_, x_split, y_, y_split = train_test_split(x_, y_, test_size=(1.0 / (fold + 1)))
        x_data.append(x_split)
        y_data.append(y_split)
    x_data.append(x_)
    y_data.append(y_)
    del x_, y_, x_split, y_split
    
    w_list = []
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    # begin trainning
    # 10 times training
    # cause each training is independent, using multiprocess
    lock = multiprocessing.Manager().Lock()
    queue = multiprocessing.Queue()
    task = [0] * 10
    for fold in range(10):
        task[fold] = multiprocessing.Process(target=training, args=(fold, x_data, y_data, queue, lock, ))
        task[fold].start()
    # collect data
    while(not queue.empty() or len(w_list) < 10):
        ret = queue.get()
        w_list.append(ret[0])
        tprs.append(ret[1])
        aucs.append(ret[2])
    # join process
    for fold in range(10):
        task[fold].join()
#         # original training
#     for fold in range(10):
#         tmp = list(range(fold)) + list(range(fold+1,10))
#         x_test = x_data[fold]
#         y_test = y_data[fold]
#         x_train = x_data[tmp[0]]
#         y_train = y_data[tmp[0]]
#         for idx in range(1, 9):
#             x_train = np.concatenate((x_train, x_data[idx]))
#             y_train = np.concatenate((y_train, y_data[idx]))
#         # train
#         w = onlineAucMaximum(x_train, y_train, N_BUFF_POS, N_BUFF_NEG, C)
#         w_list.append(w)
#         # predict
#         y_pre = np.dot(x_test, w)
#         # for auc
#         fpr, tpr, thresholds = roc_curve(y_test, y_pre)
#         roc_auc = auc(fpr, tpr)
#         tprs.append(interp(mean_fpr, fpr, tpr))
#         tprs[-1][0] = 0.0
#         aucs.append(roc_auc)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
#     std_tpr = np.std(tprs, axis=0)
#     tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
#     tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    
    print("Training AUC Score: %.3f"%(mean_auc))
    
    # no test data
    if(len(sys.argv) < 3):
        return 0
    w = np.mean(w_list, axis = 0)
    test_data = loadmat(test_path)
    x_test = test_data['x_tr']
    scaler.transform(x_test)
    y_test = test_data['y_tr']
    bia = np.ones((len(y_test), 1))
    x_test = np.hstack((x_test, bia))
    y_pre = np.dot(x_test, w)
    fpr, tpr, thresholds = roc_curve(y_test, y_pre)
    roc_auc = auc(fpr, tpr)
    print("Testing AUC Score: %.3f"%(roc_auc))
    
if __name__ == "__main__":
    main()