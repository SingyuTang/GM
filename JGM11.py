import os

import numpy as np
from matplotlib import pyplot as plt
import math
plt.rcParams['font.sans-serif']=['SimHei']#显示中文标签
plt.rcParams['axes.unicode_minus']=False
datafilepath=r'./data.txt'
def JGM11_readdata(filepath:str):
    data=np.loadtxt(filepath)
    return data
def JGM11_1(data:np.ndarray,k):
    '''
        利用累加，计算一个点的预测值
        :param data:
        :param k:
        :param alpha:
        :return:
        '''
    n = len(data)
    x0 = data.reshape(n, 1)
    x1_list = []
    tmp = 0
    for x in list(x0):
        tmp = x + tmp
        x1_list.append(tmp)
    print('x1:',x1_list)
    x1 = np.asarray(x1_list, dtype='f8')
    zlist = []
    for index in range(n - 2):
        x_half=x1[index]+0.5*(x1[index+2]-x1[index+1])
        print('xhalf:',x_half)
        ztmp=0.25*x1[index]+0.5*x1[index+1]+0.25*x_half
        zlist.append(ztmp)
    z = np.asarray(zlist).reshape(n - 2, 1)#z比x1少两个元素，故B矩阵为（n-2）*2，Y矩阵也要舍去前两个元素
    print('z:',z)
    one = np.ones([n - 2, 1], dtype='f8')
    B = np.hstack((-z, one))
    print('B:',B)
    Y = x0[1:-1].reshape(n - 2, 1)
    print('Y:',Y)
    u1 = np.linalg.inv(np.dot(B.T, B))
    u2 = np.dot(u1, B.T)
    u = np.dot(u2, Y)
    a, b = u[0], u[1]
    # print('u:',u)
    # print('a:',a,'b:',b)
    if k > 1:
        predict_xk = (1-math.exp(a))*(x0[0]-b/a)*math.exp(-a*(k-1))
        # print('front:',x0[0]-b/a)
    elif k == 1:
        predict_xk = x0[0]
    else:
        predict_xk = False
        print('请输入的k值为大于0的正整数')
    return predict_xk
data=JGM11_readdata(datafilepath)
xk=JGM11_1(data,12)
print(xk)