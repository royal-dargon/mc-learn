from numpy import *
import matplotlib.pylab as plt

# 在这个函数中实现了对text文件中的数据的提取
def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split()
        for i in range(2):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[2]))
    return dataMat, labelMat


def standRegres(xArr, yArr):
    # 转换成矩阵
    xMat = mat(xArr)
    yMat = mat(yArr).T

    # 这里是求矩阵的逆矩阵
    # w = （XTX）的逆乘上X的转置乘上y
    xTx = xMat.T * xMat     # 矩阵的转置乘上了本身
    if linalg.det(xTx) == 0:
        return
    ws = xTx.I * (xMat.T * yMat)   # .I是逆
    return ws


def regression1():
    xArr, yArr = loadDataSet("data.txt")
    xMat = mat(xArr)
    yMat = mat(yArr)
    ws = standRegres(xArr, yArr)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:, 1].flatten().tolist(), yMat.T[:, 0].flatten().A[0].tolist())
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws
    ax.plot(xCopy[:, 1], yHat)
    plt.show()


if __name__ == "__main__":
    regression1()