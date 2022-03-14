##利用GM(1,1)累加法计算预测值
import numpy as np
import math
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']#显示中文标签
plt.rcParams['axes.unicode_minus']=False
datafilepath=r'./data.txt'
def gm11_0_readdata(filepath:str):
    data=np.loadtxt(filepath)
    return data

def gm11_1(data:np.ndarray,k,alpha=0.5):
    '''
    利用累加，计算一个点的预测值
    :param data:
    :param k:
    :param alpha:
    :return:
    '''
    n=len(data)
    x0=data.reshape(n,1)
    x1_list=[]
    tmp=0
    for x in list(x0):
        tmp=x+tmp
        x1_list.append(tmp)
    # print('x1:',x1_list)
    x1=np.asarray(x1_list,dtype='f8')
    zlist=[]
    for index in range(n-1):
        ztmp=x1[index]*alpha+x1[index+1]*(1-alpha)
        zlist.append(ztmp)
    z=np.asarray(zlist).reshape(n-1,1)
    # print('z:',z)
    one=np.ones([n-1,1],dtype='f8')
    B=np.hstack((-z,one))
    # print('B:',B)
    Y=x0[1:].reshape(n-1,1)
    # print('Y:',Y)
    u1 = np.linalg.inv(np.dot(B.T, B))
    u2 = np.dot(u1, B.T)
    u = np.dot(u2, Y)
    a,b=u[0],u[1]
    # print('u:',u)
    # print('a:',a,'b:',b)
    if k>1:
        predict_xk=(x0[0]-b/a)*math.exp(-a*(k-1))+b/a-((x0[0]-b/a)*math.exp(-a*(k-2))+b/a)
        # print('front:',x0[0]-b/a)
    elif k==1:
        predict_xk=x0[0]
    else:
        predict_xk=False
        print('请输入的k值为大于0的正整数')
    return predict_xk

def gm11_2(data:np.ndarray,kmax:int,alpha=0.5):
    '''
    利用累加计算k个点的预测值
    :param data:
    :param kmax:
    :param alpha:
    :return:
    '''
    xklist=[]
    for i in range(1,kmax+1,1):
        xi=gm11_1(data,i)
        xklist.append(xi)
    xk_np=np.asarray(xklist)
    return xk_np

def gm11_main(filepath:str):
    data1=gm11_0_readdata(filepath)
    data2=gm11_2(data1,len(data1)+2)
    # print(len(data1),data1)
    print(len(data2),data2)
    plt.plot(data1)
    plt.plot(data2)
    plt.legend(['实际数据','GM(1,1)预测数据'],loc='best')
    plt.show()

gm11_main(datafilepath)



