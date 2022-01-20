# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 14:52:51 2021

@author: 25326
"""

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from seperated_classifier import CNNet,CNNet1,BPNNet,RBFNet,Auto_Encode
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import scale,StandardScaler
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from dimensionality_reduction_com import pca,kernelPCA,factorAnalysis
import keras.utils as utils
from keras.metrics import *

start = time.time()

def compute_pos_neg(y_actual, y_hat):
    TP = 0; FP = 0;TN = 0; FN = 0
    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1
    return TP,FP,TN,FN
def metrics(TP,FP,TN,FN):
    a=TP+FP
    b=TP+FN
    c=TN+FP
    d=TN+FN
    mcc=((TP*TN)-(FP*FN))/(np.sqrt(float(a*b*c*d)+0.0001))
    #F1=(2*TP)/float(2*TP+FP+FN+.0000001)
    # precision=TP/float(TP+FP+.0000001)
    #recall=TP/float(TP+FN+.0000001)
    return mcc

#生成输入数据
data_train = sio.loadmat(r'D:\Python\My__python\蛋白质相互作用\Ensemble_methods\Laplace_human.mat')
name1=list(data_train.keys())[-1]
data=data_train.get(name1)#obtain the data
row=data.shape[0]
column=data.shape[1]
index = [i for i in range(row)]
np.random.shuffle(index)
index=np.array(index)
shu=data[:,np.array(range(1,column))]
# shu=scale(shu)
label=data[:,0]

# encode=Auto_Encode(1960, 300)
# encode.fit(shu, n_epoch=800)
# data_2=encode.Encode(shu)
# shu=data_2

#data_2=pca(shu,percentage=9.5)
#shu=data_2

data_2=factorAnalysis(shu,percentage=0.99)#The rate of contribution
shu=data_2

X=shu[index,:]
y=label[index]



#模型初始参数设置
params5 = {'in_dim':X.shape[1], 'n_hidden_1':256, 'n_hidden_2':256, 'out_dim':2}
model5 = BPNNet(**params5)

sepscores = []
ytest=np.ones((1,2))*0.5
yscore=np.ones((1,2))*0.5
#K-折交叉验证与训练
skf= StratifiedKFold(n_splits=5)
for train, test in skf.split(X,y):
    print(test)
skf=list(skf.split(X,y))

S=[]
for i in range(len(skf)):
    train=skf[i][0];test=skf[i][1]
    x_train=X[train]
    y_train=y[train]
    x_test=X[test]
    y_test=y[test]
    
    Model=[]
    model=[]
    for i in range(1):
        print('第%02d'%(i+1),'个神经网络！')
        model5,Loss5,Score5,Model5=model5.fit(x_train, y_train, n_batches=1, init=True,
                                       n_epoch=2500, X_t=X[test], y_t=y[test])
        model.append(model5)
        Model+=Model5
    
    x_test1=X[test]
    y_test1=y[test]
    
    y_score5=model5.predict_proba(x_test1)
    
    y_test2=utils.to_categorical(y_test1)
    # fpr3, tpr3, _ = roc_curve(y_test2[:,0], y_score3[:,0])
    # fpr4, tpr4, _ = roc_curve(y_test2[:,0], y_score4[:,0])
    fpr5, tpr5, _ = roc_curve(y_test2[:,0], y_score5[:,0])
    # fpr8, tpr8, _ = roc_curve(y_test2[:,0], y_score8[:,0])
    # fpr, tpr, _ = roc_curve(y_test2[:,0], y_score[:,0])
    # roc_auc3 = auc(fpr3, tpr3)
    # roc_auc4 = auc(fpr4, tpr4)
    roc_auc5 = auc(fpr5, tpr5)
    # roc_auc8 = auc(fpr8, tpr8)
    # roc_auc = auc(fpr, tpr)
    p5, r5, _ = precision_recall_curve(y_test2[:,0], y_score5[:,0])
    
    y_class5 = np.argmax(y_score5,axis=1)
    
    
    m = Accuracy()
    m.update_state(y_test1, y_class5); acc5 = m.result().numpy(); m.reset_states()
    
    m = Precision()
    m.update_state(y_test1, y_class5); precision5 = m.result().numpy(); m.reset_states()
    
    m = Recall()
    m.update_state(y_test1, y_class5); recall5 = m.result().numpy(); m.reset_states()
    
    TP,FP,TN,FN = compute_pos_neg(y_test, y_class5)
    Mcc = metrics(TP,FP,TN,FN)
    S.append([acc5,precision5,recall5,Mcc])
S=np.array(S)

fig=plt.figure("ROC")
ax1=fig.add_subplot(1,1,1)
ax1.plot(fpr5,tpr5,label='BPNN')
ax1.set_title('Receiver operating characteristic')
ax1.legend(loc='lower right')
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_xlim(0,1)
ax1.set_ylim(0,1)

fig2=plt.figure("P-R Curve")
ax2=fig2.add_subplot(1,1,1)
ax2.set_title('Precision/Recall Curve')
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_xlim(0,1)
ax2.set_ylim(0,1)
ax2.plot(r5,p5,label='BPNN')
ax2.legend(loc='lower right')

#%% 
MODEL=model
y_Score=[]
y_Class=[]
y_Entropy=[]
y_Error=[]
for i in range(len(MODEL)):
    y_score_=MODEL[i].predict_proba(x_test1)
    # y_score_=np.exp(y_score_)/np.sum(np.exp(y_score_),1).reshape((-1,1))
    y_score_=y_score_/np.sum(y_score_,1).reshape((-1,1))
    entropy_=np.sum(-y_score_*np.log(y_score_),1)#caculating the Information entropy
    y_class_=np.argmax(y_score_,axis=1)
    y_Score.append(y_score_)
    y_Entropy.append(entropy_)
    y_Class.append(y_class_)
    y_Error.append(np.abs(y_class_-y_test1))
    m = Accuracy()
    m.update_state(y_test1, y_class_); acc5 = m.result().numpy(); m.reset_states()
    print(acc5)

fig=plt.figure()
ax1=fig.add_subplot(1,1,1)
ax1.plot(y_Error[2])
ax1.plot(y_Entropy[2])
########################相对多数集成###########################
y_class=0
for i in range(len(MODEL)):
    y_class+=y_Class[i]

def f(x):
    if x>len(MODEL)/2.:
        return 1
    else:
        return 0
y_class=np.array(list(map(f,y_class)))
m = Accuracy()
m.update_state(y_test1, y_class); acc = m.result().numpy(); m.reset_states()
print(acc)
###########################不确定程度集成##################################
y_entropy=0
for i in range(len(MODEL)):
    y_entropy+=1/y_Entropy[i]
y_score=0
for i in range(len(MODEL)):
    y_score+=y_Score[i]*(1/y_Entropy[i]/y_entropy).reshape((-1,1))
y_class=np.argmax(y_score,axis=1)
m = Accuracy()
m.update_state(y_test1, y_class); acc = m.result().numpy(); m.reset_states()
print(acc)
#################################################

#%% test for independent datasets
import numpy as np
import pandas as pd
#from time import time
import time
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from seperated_classifier import CNNet,CNNet1,BPNNet,WSRC_proba,Auto_Encode
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import scale,StandardScaler
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from dimensionality_reduction_com import pca,kernelPCA,factorAnalysis
# from sklearn.manifold import TSNE
import keras.utils as utils
from keras.metrics import *

start = time.time()
 
def compute_pos_neg(y_actual, y_hat):
    TP = 0; FP = 0;TN = 0; FN = 0
    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1
    return TP,FP,TN,FN
def metrics(TP,FP,TN,FN):
    a=TP+FP
    b=TP+FN
    c=TN+FP
    d=TN+FN
    mcc=((TP*TN)-(FP*FN))/(np.sqrt(float(a*b*c*d)+0.0001))
    #F1=(2*TP)/float(2*TP+FP+FN+.0000001)
    # precision=TP/float(TP+FP+.0000001)
    #recall=TP/float(TP+FN+.0000001)
    return mcc

#生成输入数据
data_train = sio.loadmat(r'Laplace_human.mat')
name1=list(data_train.keys())[-1]
data=data_train.get(name1)#obtain the data
# data_train = sio.loadmat(r'Laplace_pylori.mat')
# name1=list(data_train.keys())[-1]
# data1=data_train.get(name1)#obtain the data
# data=np.concatenate((data,data1),axis=0)
l1=data.shape[0]

S=[]
for name in ['Celeg_Laplace.mat','Ecoli_Laplace.mat','Hsapi_Laplace.mat','Mmusc_Laplace.mat']:
    data_train = sio.loadmat(r'%s'%name)
    name1=list(data_train.keys())[-1]
    data1=data_train.get(name1)#obtain the data
    l2=data1.shape[0]
    
    data2=np.concatenate((data,data1),axis=0)
    row=data2.shape[0]
    column=data2.shape[1]
    index = [i for i in range(row)]
    # np.random.shuffle(index)
    index=np.array(index)
    shu=data2[:,np.array(range(1,column))]
    # shu=scale(shu)
    label=data2[:,0]
    
    # encode=Auto_Encode(1960, 700)
    # encode.fit(shu, n_epoch=1000)
    # data_2=encode.Encode(shu)
    # shu=data_2
    
    # data_2=pca(shu,percentage=0.99)
    # shu=data_2
    
    # data_2,mask=logistic_dimension(shu,label,parameter=0.05)
    # shu=data_2
    
    data_2=factorAnalysis(shu,percentage=0.99)#The rate of contribution
    shu=data_2
    
    X=shu[index,:]
    y=label[index]
    
    train=np.arange(l1)
    test=np.arange(l1,l1+l2)
    x_train=X[train]
    y_train=y[train]
    x_test=X[test]
    y_test=y[test]
    
    Model=[]
    model=[]
    for i in range(1):
        print('第%02d'%(i+1),'个神经网络！')
        params5 = {'in_dim':X.shape[1], 'n_hidden_1':256, 'n_hidden_2':256, 'out_dim':2}
        model5 = BPNNet(**params5)
        model5,Loss5,Score5,Model5,R=model5.fit(x_train, y_train, n_batches=1, init=True,
                                        n_epoch=2500, X_t=X[test], y_t=y[test])
        model.append(model5)
        Model+=Model5
    
    # model5 = GradientBoostingClassifier()
    # model5=model5.fit(x_train, y_train)
    
    y_score5=model5.predict_proba(x_test)
    
    y_test2=np.column_stack((1-y_test.reshape(-1,1),y_test.reshape(-1,1)))
    # fpr3, tpr3, _ = roc_curve(y_test2[:,0], y_score3[:,0])
    # fpr4, tpr4, _ = roc_curve(y_test2[:,0], y_score4[:,0])
    fpr5, tpr5, _ = roc_curve(y_test2[:,0], y_score5[:,0],pos_label=1)
    # fpr8, tpr8, _ = roc_curve(y_test2[:,0], y_score8[:,0])
    # fpr, tpr, _ = roc_curve(y_test2[:,0], y_score[:,0])
    # roc_auc3 = auc(fpr3, tpr3)
    # roc_auc4 = auc(fpr4, tpr4)
    roc_auc5 = auc(fpr5, tpr5)
    # roc_auc8 = auc(fpr8, tpr8)
    # roc_auc = auc(fpr, tpr)
    p5, r5, _ = precision_recall_curve(y_test2[:,0], y_score5[:,0])
    
    y_class5 = np.argmax(y_score5,axis=1)
    
    
    m = Accuracy()
    m.update_state(y_test, y_class5); acc5 = m.result().numpy(); m.reset_states()
    
    m = Precision()
    m.update_state(y_test, y_class5); precision5 = m.result().numpy(); m.reset_states()
    
    m = Recall()
    m.update_state(y_test, y_class5); recall5 = m.result().numpy(); m.reset_states()
    
    TP,FP,TN,FN = compute_pos_neg(y_test, y_class5)
    Mcc = metrics(TP,FP,TN,FN)
    S.append([acc5,precision5,recall5,Mcc])
S=np.array(S)

fig=plt.figure("ROC")
ax1=fig.add_subplot(1,1,1)
ax1.plot(fpr5,tpr5,label='BPNN')
ax1.set_title('Receiver operating characteristic')
ax1.legend(loc='lower right')
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_xlim(0,1)
ax1.set_ylim(0,1)

fig2=plt.figure("P-R Curve")
ax2=fig2.add_subplot(1,1,1)
ax2.set_title('Precision/Recall Curve')
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_xlim(0,1)
ax2.set_ylim(0,1)
ax2.plot(r5,p5,label='BPNN')
ax2.legend(loc='lower right')


