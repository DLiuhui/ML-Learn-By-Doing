import numpy as np
import operator # 运算符模块 operator module

# create random dataset
def creatRandomDataSet(n_data, n_features):
    n_data_1 = int(n_data * np.random.rand())
    n_data_2 = int(n_data - n_data_1)
    group_1 = np.random.rand(n_data_1, n_features) * 10 + 1
    labels_1 = np.ones(n_data_1)
    group_2 = np.random.rand(n_data_2, n_features) * 30 + 20
    labels_2 = np.zeros(n_data_2)
    group = np.vstack((group_1, group_2))
    labels = np.hstack((labels_1, labels_2))
    return group, labels

# 分类器
def classify0(X, dataset, labels, k):
    n_dataset = len(dataset)
    # np.tile函数用于制作向量
    diff_mat = np.tile(X, (n_dataset, 1)) - dataset # 计算距离差
    sq_diff_mat = diff_mat ** 2 # 平方
    sq_distance = sq_diff_mat.sum(axis=1)
    distance = sq_distance ** 0.5
    sorted_distance_idx = np.argsort(distance)
    class_count = {}
    for i in range(k): # 判断最接近的k个点的类型,并进行统计
        vote_label = labels[sorted_distance_idx[i]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    return max(class_count, key=lambda x: class_count[x])

# reading data from .txt file
# As a preparation, should first change string labels into integer numbers
def file2matrix(filename, n_features, delimiter='\t'):
    if(not n_features > 0):
        print('error, features should > 0')
        return
    file = open(filename)
    lines = file.readlines()
    numberOfLines = len(lines)
    ret_mat = np.zeros((numberOfLines, n_features))
    labels = np.zeros(numberOfLines)
    index = 0
    for line in lines:
        line = line.strip()
        list_from_line = line.split(delimiter)
        ret_mat[index,:] = list_from_line[0:n_features]
        labels[index] = int(list_from_line[-1])
        index += 1
    return ret_mat, labels

# 归一化
def autoNorm(dataset):
    minvals = dataset.min(axis=0)
    maxvals = dataset.max(axis=0)
    ranges = maxvals - minvals
    total = dataset.shape[0]
    norm_dataset = dataset - np.tile(minvals, (total, 1))
    norm_dataset = norm_dataset / np.tile(ranges, (total, 1))
    return norm_dataset, ranges, minvals

# knn用于约会网站数据匹配
# testing code 测试代码
# 交叉检验
def datingClassTest():
    ratio = 0.10
    dating_data, dating_labels = file2matrix('datingTestSet2.txt', 3)
    dating_data, ranges, minvals = autoNorm(dating_data)
    total = dating_data.shape[0]
    n_test = int(total * ratio)
    n_error = 0
    for i in range(n_test):
        classify_result = classify0(dating_data[i, :], dating_data[n_test:,:], dating_labels[n_test:], 3)
        print('the classifier came back with: %d, the number is: %d'%(classify_result, dating_labels[i]))
        if(classify_result != dating_labels[i]):
            n_error += 1
    print('the total error rate is: %.3f'%(n_error/total))

# 实际使用
# 外部输入数据
def classifyPerson():
    result_list = ['not at all', 'in small doses', 'in large doses']
    percent_tats = float(input('percentage of time spent playing video games?'))
    ff_miles = float(input('frequent flier miles earned per year?'))
    ice_cream = float(input('liters of ice cream consumed per year?'))
    dataset, labels = file2matrix('datingTestSet2.txt', 3)
    dataset, ranges, minvals = autoNorm(dataset)
    test_data = np.array([ff_miles, percent_tats, ice_cream])
    test_data = (test_data - minvals) / ranges
    result = int(classify0(test_data, dataset, labels, 3))
    print('You will probably like this person:', result_list[result - 1])