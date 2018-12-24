import knn
import numpy as np
import os

def img2vector(filename):
    ret_vector = np.zeros((1, 1024)) # 将32*32二进制图像转换成1*1024行向量
    file = open(filename)
    for i in range(32):
        line = file.readline()
        for j in range(32):
            ret_vector[0, i*32 + j] = int(line[j])
    return ret_vector

# 使用knn进行手写识别
def handWritingClassTest():
    training_file_list = os.listdir('trainingDigits')
    n_train = len(training_file_list)
    train_labels = np.zeros(n_train)
    training_mat = np.zeros((n_train, 1024))
    for i in range(n_train):
        filename = training_file_list[i]
        training_mat[i,:] = img2vector('trainingDigits/%s'%filename)
        train_labels[i] = int(filename[0])
    # test
    test_file_list = os.listdir('testDigits')
    n_test = len(test_file_list)
    n_error = 0
    for i in range(n_test):
        filename = test_file_list[i]
        test_vector = img2vector('testDigits/%s' % filename)
        test_label = int(filename[0])
        classify_label = knn.classify0(test_vector, training_mat, train_labels, 3)
        print('the classifier came back with %d, the real number = %d'%(classify_label, test_label))
        if(classify_label != test_label):
            n_error += 1
    print('the total number of errors is: %d'%n_error)
    print('the total error rate is: %.5f' %(n_error/n_test))

if __name__=='__main__':
    handWritingClassTest()