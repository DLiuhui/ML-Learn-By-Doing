import knn
import matplotlib.pyplot as plt

if __name__ == '__main__':
    x,y = knn.file2matrix('datingTestSet2.txt', n_features=3)
    x, ranges, minvals = knn.autoNorm(x)
    # plot
    plt.figure(1)
    plt.scatter(x[:,1], x[:,2], 15.0 * y, 15.0 * y)
    plt.show()
    knn.classifyPerson()
