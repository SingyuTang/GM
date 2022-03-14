from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import numpy as np
import  matplotlib.pyplot  as plt
from datetime import date,datetime

import pandas
from statsmodels import api as sm

pointfilepath=r'./point/p1415_970.txt'
plt.rcParams['font.sans-serif']=['SimHei']#显示中文标签
plt.rcParams['axes.unicode_minus']=False
def read_pointfile(filepath):
    '''
    读取时序点文件中的日期和形变值并返回
    :param filepath: 文件路径
    :return: 返回日期和形变数据组成的元组，（日期，形变）。形变（deform）为ndarray类型，里面的元素为float64类型；日期为一个列表，里面的元素为datetime类型
    '''
    data=np.genfromtxt(filepath,skip_header=9,delimiter='',dtype='U20')#dtype为浮点不能读取时间（字符串）
    # print(type(data),data)
    datearray,deform=data[...,1],data[...,4]
    deform=deform.astype('f8')#更改array数据类型为float64
    datelist=list(datearray)
    YMDlist=[]
    for tmp in datelist:
        YMDs=tmp[0:10]
        YMDt=datetime.strptime(YMDs,'%Y-%m-%d')
        YMDlist.append(YMDt)

    # print(type(deform[1])
    return YMDlist,deform
def draw_diff(deform:np.ndarray,suptitle='差分图'):
    '''
    绘制差分图
    :param deform: 由read_pointfile中读取的deform：np.ndarray
    :return:
    '''
    deform_diff1=np.diff(deform,axis=0,n=1)#axis=0：行相减，axis=1：列相减
    deform_diff2=np.diff(deform,axis=0,n=2)
    deform_diff3=np.diff(deform,axis=0,n=3)
    # print(date)
    fig0=plt.figure(figsize=(8,6))
    ax_diff0=fig0.add_subplot(221)
    plt.plot(deform)
    ax_diff0.xaxis.set_ticks_position('bottom')
    # fig0.tight_layout()
    plt.title('原始图')
    ax_diff1=fig0.add_subplot(222)
    plt.plot(deform_diff1)
    ax_diff1.xaxis.set_ticks_position('bottom')
    plt.title('一阶差分图')
    ax_diff2=fig0.add_subplot(223)
    plt.plot(deform_diff2)
    ax_diff2.xaxis.set_ticks_position('bottom')
    plt.title('二阶差分图')
    ax_diff3=fig0.add_subplot(224)
    plt.plot(deform_diff3)
    ax_diff3.xaxis.set_ticks_position('bottom')
    plt.title('三阶差分图')
    plt.suptitle(suptitle)
    plt.show()
    plt.close()
def draw_acf_pacf(deform:np.ndarray,n_diff=0,suptitle='ACF AND PACF'):
    '''
    绘制deform的acf和pacf图，用来查找AR（p）和MA（q）
    :param deform: read_pointfile中读取文本返回的数据
    :param n_diff: n阶差分
    :return:
    '''
    deform_diff = np.diff(deform, axis=0, n=n_diff)  # axis=0：行相减，axis=1：列相减
    print(deform_diff)
    n_acf=len(deform_diff)-1
    n_pacf=(n_acf/2)-1
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(211)
    fig = plot_acf(deform_diff, lags=n_acf, ax=ax1)
    ax1.xaxis.set_ticks_position('bottom')
    ax2 = fig.add_subplot(212)
    fig = plot_pacf(deform_diff, lags=n_pacf, ax=ax2,method='ywm')
    ax2.xaxis.set_ticks_position('bottom')
    plt.suptitle(suptitle)
    plt.show()
    plt.close()



# pointdata=read_pointfile(pointfilepath)
# date,deform=pointdata[0],pointdata[1]
# draw_diff(deform,'p1415_970差分图')
# draw_acf_pacf(deform,0)
