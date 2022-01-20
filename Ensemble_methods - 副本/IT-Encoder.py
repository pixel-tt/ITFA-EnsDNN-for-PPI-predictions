# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 15:34:38 2021

@author: 25326
"""

import re
import numpy as np
import scipy.io as sio


#%% function
def Laplace(ss,N,composition):
    while re.search('X',ss) or re.search('U',ss):
        ss = re.sub('X','',ss)
        ss = re.sub('U','',ss)
    l=len(ss)
    tt='ACDEFGHIKLMNPQRSTVWY'
    #氨基酸的7种物理化学性质对应的数值
    # "hydrophobicity","hydrophilicity","volume of side
    # chains","polarity","polarizability","solvent accessible surface
    # area","net charge index","mass of side chains"
    property_=np.array([[0.620000000000000,0.290000000000000,-0.900000000000000,-0.740000000000000,1.19000000000000,0.480000000000000,-0.400000000000000,1.38000000000000,-1.50000000000000,1.06000000000000,0.640000000000000,-0.780000000000000,0.120000000000000,-0.850000000000000,-2.53000000000000,-0.180000000000000,-0.0500000000000000,1.08000000000000,0.810000000000000,0.260000000000000],
        [-0.500000000000000,-1,3,3,-2.50000000000000,0,-0.500000000000000,-1.80000000000000,3,-1.80000000000000,-1.30000000000000,2,0,0.200000000000000,3,0.300000000000000,-0.400000000000000,-1.50000000000000,-3.40000000000000,-2.30000000000000],
        [27.5000000000000,44.6000000000000,40,62,115.500000000000,0,79,93.5000000000000,100,93.5000000000000,94.1000000000000,58.7000000000000,41.9000000000000,80.7000000000000,105,29.3000000000000,51.3000000000000,71.5000000000000,145.500000000000,117.300000000000],
        [8.10000000000000,5.50000000000000,13,12.3000000000000,5.20000000000000,9,10.4000000000000,5.20000000000000,11.3000000000000,4.90000000000000,5.70000000000000,11.6000000000000,8,10.5000000000000,10.5000000000000,9.20000000000000,8.60000000000000,5.90000000000000,5.40000000000000,6.20000000000000],
        [0.0460000000000000,0.128000000000000,0.105000000000000,0.151000000000000,0.290000000000000,0,0.230000000000000,0.186000000000000,0.219000000000000,0.186000000000000,0.221000000000000,0.134000000000000,0.131000000000000,0.180000000000000,0.291000000000000,0.0620000000000000,0.108000000000000,0.140000000000000,0.409000000000000,0.298000000000000],
        [1.18100000000000,1.46100000000000,1.58700000000000,1.86200000000000,2.22800000000000,0.881000000000000,2.02500000000000,1.81000000000000,2.25800000000000,1.93100000000000,2.03400000000000,1.65500000000000,1.46800000000000,1.93200000000000,2.56000000000000,1.29800000000000,1.52500000000000,1.64500000000000,2.66300000000000,2.36800000000000],
        [0.00718700000000000,-0.0366100000000000,-0.0238200000000000,0.00680200000000000,0.0375500000000000,0.179100000000000,-0.0106900000000000,0.0216300000000000,0.0177100000000000,0.0516700000000000,0.00268300000000000,0.00539200000000000,0.239500000000000,0.0492100000000000,0.0435900000000000,0.00462700000000000,0.00335200000000000,0.0570000000000000,0.0379800000000000,0.0236000000000000],
        [15,47,59,73,91,1,82,57,73,57,75,58,42,72,101,31,45,43,130,107]])
    #Normalization
    property_=(property_-property_.mean(1).reshape(-1,1))/property_.std(1).reshape(-1,1)
    
    property1=property_
    
    data=np.zeros(l)
    f=np.zeros((1,20))
    for j in range(l):
        for k in range(20):
            if ss[j]==tt[k]:
                data[j]=k; f[0,k]+=1
    f=f/l
    DATA=property1[:,data.astype('int')]
    if l<N:
        DATA=np.concatenate((DATA,np.zeros((8,N-l))), axis=1)
        l=DATA.shape[1]

    # 离散指数函数卷积（离散积分变换编码）
    a=1
    result=[]
    for x in range(N):
        r=np.sum(DATA[:,x:]*np.exp(np.array([range(l-x) for i in range(8)])*(-a)), 
                 axis=1)*(1-np.exp(-a))#设置加权参数
        result.append(r)
    
    # 离散傅里叶变换（离散积分变换编码）
    for x in range(N):
        r=np.sum(DATA*np.exp(np.array([range(l) for i in range(8)])*(-2*np.pi*1j)/N*x), axis=1)#设置加权参数
        result.append(np.real(r))
    
    result=np.array(result)
    #加入氨基酸序列的成分信息,原始序列信息
    if composition==True:
        # vector=np.concatenate((f,DATA[:,:N].reshape(1,8*N),result.reshape(1,2*8*N)), axis=1)
        vector=np.concatenate((f, result.reshape(1,2*8*N)), axis=1)
    else:
        vector=result.reshape(1,2*8*N)
        
    return vector

#%% encoding for benchmarks data

data_train = sio.loadmat(r'./Dataset/yeast_data.mat')#load animo acid sequences 
P_protein_a=data_train.get('P_protein_a')
P_protein_b=data_train.get('P_protein_b')
N_protein_a=data_train.get('N_protein_a')
N_protein_b=data_train.get('N_protein_b')

num1=len(P_protein_a)
num2=len(N_protein_a)
result_1=[]
result_11=[]
result_2=[]
result_22=[]
N=60
for i in range(num1):
    result1=Laplace(P_protein_a[i][0][0],N,True)#正序编码
    result11=Laplace(P_protein_b[i][0][0],N,True)
    # result1=Laplace(P_protein_a[i][0][0][::-1],N,True);#逆序编码
    # result11=Laplace(P_protein_b[i][0][0][::-1],N,True)
    result_1.append(result1)
    result_11.append(result11)
result_1=np.array(result_1)
result_11=np.array(result_11)

for i in range(num2):
    result2=Laplace(N_protein_a[i][0][0],N,True)
    result22=Laplace(N_protein_b[i][0][0],N,True)
    # result2=Laplace(N_protein_a[i][0][0][::-1],N,True)
    # result22=Laplace(N_protein_b[i][0][0][::-1],N,True)
    result_2.append(result2)
    result_22.append(result22)
result_2=np.array(result_2)
result_22=np.array(result_22)

Pa=np.squeeze(result_1)
Pb=np.squeeze(result_11)
Na=np.squeeze(result_2)
Nb=np.squeeze(result_22)
P=np.concatenate((np.ones((len(P_protein_a),1)),Pa,Pb), axis=1)
N=np.concatenate((np.zeros((len(N_protein_a),1)),Na,Nb), axis=1)
data_SWA=np.concatenate((P,N), axis=0)
np.save('Laplace_yeast_{}.npy'.format(N), data_SWA)

#%% encoding for Wnt_related data

f1=open(r'./Dataset/PPIs_network/Wnt_related/Wnt_proteinA.txt').readlines()#读取氨基酸序列文件
f2=open(r'./Dataset/PPIs_network/Wnt_related/Wnt_proteinB.txt').readlines()
P_protein_a=[]
P_protein_b=[]
for i in range(len(f1)):
    if re.match('>',f1[i]):
        j=1
        a=''
        while not re.match('>',f1[i+j]):
            a=a+f1[i+j]
            j+=1
            if i+j>=len(f1):
                break
        # if a=='':
        #     continue
        a=re.sub('\n','',a)
        P_protein_a.append(a)
    if re.match('>',f2[i]):
        j=1
        a=''
        while not re.match('>',f2[i+j]):
            a=a+f2[i+j]
            j+=1
            if i+j>=len(f2):
                break
        # if a=='':
        #     continue
        a=re.sub('\n','',a)
        P_protein_b.append(a)


num1=len(P_protein_a)
result_1=[]
result_11=[]
N=60
for i in range(num1):
    result1=Laplace(P_protein_a[i],N,True)#正序编码
#    result1_inv=Laplace(P_protein_a[i][::-1],N,False);%逆序编码
    result11=Laplace(P_protein_b[i],N,True)
#    result11_inv=Laplace(P_protein_b[i][::-1],N,False)
    result_1.append(result1)
    result_11.append(result11)
result_1=np.array(result_1)
result_11=np.array(result_11)


Pa=np.squeeze(result_1)
Pb=np.squeeze(result_11)
P=np.concatenate((np.ones((len(P_protein_a),1)),Pa,Pb), axis=1)
data_SWA=P
np.save('wnt.npy', data_SWA)

#%% encoding for one_core net data

f1=open(r'./Dataset/PPIs_network/one_core/sequence.fasta').readlines()#读取氨基酸序列文件
P_protein_a=[]
P_protein_b=[]
for i in range(len(f1)):
    if re.match('>',f1[i]):
        j=1
        a=''
        while not re.match('>',f1[i+j]):
            a=a+f1[i+j]
            j+=1
            if i+j>=len(f1):
                break
        # if a=='':
        #     continue
        a=re.sub('\n','',a)
        break
for i in range(1,len(f1)):
    if re.match('>',f1[i]):
        j=1
        a=''
        while not re.match('>',f1[i+j]):
            a=a+f1[i+j]
            j+=1
            if i+j>=len(f1):
                break
        # if a=='':
        #     continue
        a=re.sub('\n','',a)
        P_protein_a.append(a)
        P_protein_b.append(a)


num1=len(P_protein_a)
result_1=[]
result_11=[]
N=60
for i in range(num1):
    result1=Laplace(P_protein_a[i],N,True)#正序编码
#    result1_inv=Laplace(P_protein_a[i][::-1],N,False);%逆序编码
    result11=Laplace(P_protein_b[i],N,True)
#    result11_inv=Laplace(P_protein_b[i][::-1],N,False)
    result_1.append(result1)
    result_11.append(result11)
result_1=np.array(result_1)
result_11=np.array(result_11)


Pa=np.squeeze(result_1)
Pb=np.squeeze(result_11)
P=np.concatenate((np.ones((len(P_protein_a),1)),Pa,Pb), axis=1)
data_SWA=P
np.save('cd9.npy', data_SWA)