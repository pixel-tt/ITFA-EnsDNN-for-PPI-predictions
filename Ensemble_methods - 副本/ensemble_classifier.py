# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 15:43:49 2020

@author: 25326
"""

import numpy as np
import pandas as pd
#from time import time
import time
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import scipy.io as sio
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from seperated_classifier import CNNet,CNNet1,BPNNet,RBFNet,Ensemble
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import scale,StandardScaler
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from dimensionality_reduction_com import pca, kernelPCA, factorAnalysis
import keras.utils as utils
from keras.metrics import *
import pickle

start = time.time()

#生成输入数据
# data_train = sio.loadmat(r'D:\Python\My__python\蛋白质相互作用\Ensemble_methods\data_ensem.mat')
# data_train = sio.loadmat(r'D:\Python\My__python\蛋白质相互作用\Ensemble_methods\PAAC_Yeast_11.mat')
data_train = sio.loadmat(r'D:\Python\My__python\蛋白质相互作用\Ensemble_methods\Laplace_yeast.mat')
# data=data_train.get('data_ensem')#obtain the data
# data=data_train.get('data_paac')
data=data_train.get('data_SWA')
row=data.shape[0] 
column=data.shape[1]
index = [i for i in range(row)]
np.random.shuffle(index)
index=np.array(index)
shu=data[:,np.array(range(1,column))]
#shu=scale(shu)
label=data[:,0]

#数据降维
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
params1 = {'kernel': 'rbf', 'probability':True}
params2 = {'loss': 'deviance', 'learning_rate': 0.1,
          'n_estimators': 1000, 'subsample': 1.0, 'criterion':'friedman_mse', 
          'min_samples_split': 2, 'min_samples_leaf': 1, 'min_weight_fraction_leaf': 0.0, 
          'max_depth': 70,'min_impurity_decrease': 0.0, 'min_impurity_split': None, 
          'max_features': 'sqrt', 'verbose': 0, 'max_leaf_nodes': None, 
          'warm_start': False, 'presort': 'auto'} #the parameters of GTB
params5 = {'in_dim':X.shape[1], 'n_hidden_1':512, 'n_hidden_2':128, 'out_dim':2}
params8 = {'samples':X, 'num_centers':104, 'dim_centure':X.shape[1]}
model1 = SVC(**params1)#using support vector machine
model2 = GradientBoostingClassifier(**params2)
model3 = CNNet()
model4 = CNNet1()
model5 = BPNNet(**params5)
model6 = KNeighborsClassifier(n_neighbors=10,weights="distance",algorithm="auto")
model8 = RandomForestClassifier(n_estimators=1000, criterion='gini', max_depth=70, 
                                min_samples_split=2, min_samples_leaf=1, 
                                min_weight_fraction_leaf=0.0, max_features='auto', 
                                max_leaf_nodes=None, min_impurity_decrease=0.0, 
                                min_impurity_split=None, bootstrap=True, 
                                oob_score=False, n_jobs=1, random_state=None, verbose=0, 
                                warm_start=False, class_weight=None)#the parameters of RF
# model7 = WSRC_proba
# model8 = RBFNet(**params8)

sepscores = []
#K-折交叉验证与训练
skf= StratifiedKFold(n_splits=5)

j=0
for train, test in skf.split(X,y):
    print('5折交叉验证：第%d次'%j)
    x_train=X[train]
    y_train=y[train]
    x_test=X[test]
    y_test=y[test]
    #模型训练（训练集）
    model1=model1.fit(x_train, y_train)
    model2=model2.fit(x_train, y_train)
    # model3.fit(x_train, y_train, n_batches=1, init=False, n_epoch=2000,
    #         X_t=X[test],y_t=y[test])
    # model4=model4.fit(x_train, y_train, n_batches=5, init=True, n_epoch=2000)
    
    Model=[]
    model=[]
    for i in range(5):
        print('第%02d'%(i+1),'个神经网络！')
        model5,Loss5,Score5,Model5=model5.fit(x_train, y_train, n_batches=1, init=True,\
                                   n_epoch=2500,X_t=X[test],y_t=y[test])
        model.append(model5)
        Model+=Model5
    model6=model6.fit(x_train, y_train)
    model8=model8.fit(x_train, y_train)
###################################Ensemble#######################################
    MODEL=Model
    y_Score=[]
    y_Class=[]
    y_Entropy=[]
    y_Error=[]
    for i in range(len(MODEL)):
        y_score_=MODEL[i].predict_proba(x_test)
        # y_score_=np.exp(y_score_)/np.sum(np.exp(y_score_),1).reshape((-1,1))
        y_score_=y_score_/np.sum(y_score_,1).reshape((-1,1))*0.99999
        entropy_=np.sum(-y_score_*np.log(y_score_),1)#caculating the Information entropy
        y_class_=np.argmax(y_score_,axis=1)
        y_Score.append(y_score_)
        y_Entropy.append(entropy_)
        y_Class.append(y_class_)
        y_Error.append(np.abs(y_class_-y_test))
        m = Accuracy()
        m.update_state(y_test, y_class_); acc5 = m.result().numpy(); m.reset_states()
        print(acc5)
    ###########################不确定程度集成##################################
    y_entropy=0
    for i in range(len(MODEL)):
        y_entropy+=1/y_Entropy[i]
    y_score=0
    for i in range(len(MODEL)):
        y_score+=y_Score[i]*(1/y_Entropy[i]/y_entropy).reshape((-1,1))
##############################################################################
        
    #模型验证（测试集）
    y_score1=model1.predict_proba(x_test)
    y_score2=model2.predict_proba(x_test)
    # y_score3=model3.predict_proba(x_test)
    # y_score4=model4.predict_proba(x_test)
    y_score5=Model[-1].predict_proba(x_test)
    y_score6=model6.predict_proba(x_test)
    # y_score7=model7(X[train],y[train],X[test])
    y_score8=model8.predict_proba(x_test)
    
    #模型性能评价
    y_test2=utils.to_categorical(y_test)
    fpr1, tpr1, _ = roc_curve(y_test2[:,0], y_score1[:,0])
    p1, r1, _ = precision_recall_curve(y_test2[:,0], y_score1[:,0])
    fpr2, tpr2, _ = roc_curve(y_test2[:,0], y_score2[:,0])
    p2, r2, _ = precision_recall_curve(y_test2[:,0], y_score2[:,0])
    # fpr3, tpr3, _ = roc_curve(y_test2[:,0], y_score3[:,0])
    # p3, r3, _ = precision_recall_curve(y_test2[:,0], y_score3[:,0])
    # fpr4, tpr4, _ = roc_curve(y_test2[:,0], y_score4[:,0])
    # p4, r4, _ = precision_recall_curve(y_test2[:,0], y_score4[:,0])
    fpr5, tpr5, _ = roc_curve(y_test2[:,0], y_score5[:,0])
    p5, r5, _ = precision_recall_curve(y_test2[:,0], y_score5[:,0])
    fpr6, tpr6, _ = roc_curve(y_test2[:,0], y_score6[:,0])
    p6, r6, _ = precision_recall_curve(y_test2[:,0], y_score6[:,0])
    fpr8, tpr8, _ = roc_curve(y_test2[:,0], y_score8[:,0])
    p8, r8, _ = precision_recall_curve(y_test2[:,0], y_score8[:,0])
    fpr, tpr, _ = roc_curve(y_test2[:,0], y_score[:,0])
    p, r, _ = precision_recall_curve(y_test2[:,0], y_score[:,0])
    roc_auc1 = auc(fpr1, tpr1)
    roc_auc2 = auc(fpr2, tpr2)
    # roc_auc3 = auc(fpr3, tpr3)
    # roc_auc4 = auc(fpr4, tpr4)
    roc_auc5 = auc(fpr5, tpr5)
    roc_auc6 = auc(fpr6, tpr6)
    roc_auc8 = auc(fpr8, tpr8)
    roc_auc = auc(fpr, tpr)
    y_class1 = np.argmax(y_score1,axis=1)
    y_class2 = np.argmax(y_score2,axis=1)
    # y_class3 = np.argmax(y_score3,axis=1)
    # y_class4 = np.argmax(y_score4,axis=1)
    y_class5 = np.argmax(y_score5,axis=1)
    y_class6 = np.argmax(y_score6,axis=1)
    y_class8 = np.argmax(y_score8,axis=1)
    y_class = np.argmax(y_score,axis=1)
    
    m = Accuracy()
    m.update_state(y_test, y_class1); acc1 = m.result().numpy(); m.reset_states()
    m.update_state(y_test, y_class2); acc2 = m.result().numpy(); m.reset_states()
    # m.update_state(y_test, y_class3); acc3 = m.result().numpy(); m.reset_states()
    # m.update_state(y_test, y_class4); acc4 = m.result().numpy(); m.reset_states()
    m.update_state(y_test, y_class5); acc5 = m.result().numpy(); m.reset_states()
    m.update_state(y_test, y_class6); acc6 = m.result().numpy(); m.reset_states()
    m.update_state(y_test, y_class8); acc8 = m.result().numpy(); m.reset_states()
    m.update_state(y_test, y_class); acc = m.result().numpy(); m.reset_states()
    
    m = Precision()
    m.update_state(y_test, y_class1); precision1 = m.result().numpy(); m.reset_states()
    m.update_state(y_test, y_class2); precision2 = m.result().numpy(); m.reset_states()
    # m.update_state(y_test, y_class3); precision3 = m.result().numpy(); m.reset_states()
    # m.update_state(y_test, y_class4); precision4 = m.result().numpy(); m.reset_states()
    m.update_state(y_test, y_class5); precision5 = m.result().numpy(); m.reset_states()
    m.update_state(y_test, y_class6); precision6 = m.result().numpy(); m.reset_states()
    m.update_state(y_test, y_class8); precision8 = m.result().numpy(); m.reset_states()
    m.update_state(y_test, y_class); precision = m.result().numpy(); m.reset_states()
    
    m = Recall()
    m.update_state(y_test, y_class1); recall1 = m.result().numpy(); m.reset_states()
    m.update_state(y_test, y_class2); recall2 = m.result().numpy(); m.reset_states()
    # m.update_state(y_test, y_class3); recall3 = m.result().numpy(); m.reset_states()
    # m.update_state(y_test, y_class4); recall4 = m.result().numpy(); m.reset_states()
    m.update_state(y_test, y_class5); recall5 = m.result().numpy(); m.reset_states()
    m.update_state(y_test, y_class6); recall6 = m.result().numpy(); m.reset_states()
    m.update_state(y_test, y_class8); recall8 = m.result().numpy(); m.reset_states()
    m.update_state(y_test, y_class); recall = m.result().numpy(); m.reset_states()


    sepscores.append([[acc1,precision1,recall1,roc_auc1],\
                      [acc2,precision2,recall2,roc_auc2],\
                          [acc5,precision5,recall5,roc_auc5],\
                              [acc6,precision6,recall6,roc_auc6],\
                                  [acc8,precision8,recall8,roc_auc8],\
                                      [acc,precision,recall,roc_auc]])
                                  
    
    j+=1
    fig=plt.figure('ROC and P-R curve', figsize=[11, 5])
    ax1=fig.add_subplot(1,2,1)
    ax1.plot(fpr1,tpr1,label='SVM')
    ax1.plot(fpr2,tpr2,label='GBT')
    # ax1.plot(fpr3,tpr3,label='CNN')
    ax1.plot(fpr5,tpr5,label='BPNN')
    ax1.plot(fpr,tpr,label='EnsBPNN')
    ax1.plot(fpr6,tpr6,label='KNN')
    ax1.plot(fpr8,tpr8,label='RF')
    ax1.set_title('Receiver operating characteristic')
    ax1.legend(loc='lower right')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_xlim(0,1)
    ax1.set_ylim(0,1)
    ax2=fig.add_subplot(1,2,2)
    ax2.plot(r1,p1,label='SVM')
    ax2.plot(r2,p2,label='GBT')
    # ax2.plot(r3,p3,label='CNN')
    ax2.plot(r5,p5,label='BPNN')
    ax2.plot(r,p,label='EnsBPNN')
    ax2.plot(r6,p6,label='KNN')
    ax2.plot(r8,p8,label='RF')
    ax2.set_title('P-R curve')
    ax2.legend(loc='lower left')
    ax2.set_xlabel('Recall(R)')
    ax2.set_ylabel('Precision(P)')
    ax2.set_xlim(0,1)
    ax2.set_ylim(0,1)
    fn='ROC_P-R(%d).png'%j
    plt.savefig(fn, dpi=150)
    plt.close(fig)
    
    # Fig[1].show()
sepscores=np.array(sepscores)
np.save('sepscores.npy',sepscores)
# np.save('sepscores1.npy',sepscores)

##############compare different Classifiers(条形图比较不同分类器的效果)####################
Sepscores=sepscores
#画图
font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)
index=np.array([1,5,9,13,17])
fig=plt.figure(figsize=[11, 8])
ax=fig.add_subplot(1,1,1)
ax.bar(index, Sepscores[:,0,0], width=0.4, label='SVM')
ax.bar(index+1*0.4, Sepscores[:,1,0], width=0.4, label='GBT')
ax.bar(index+2*0.4, Sepscores[:,2,0], width=0.4, label='BPNN')
ax.bar(index+3*0.4, Sepscores[:,3,0], width=0.4, label='KNN')
ax.bar(index+4*0.4, Sepscores[:,4,0], width=0.4, label='RF')
ax.bar(index+5*0.4, Sepscores[:,5,0], width=0.4, label='EnsBPNN')
ax.legend(loc='lower right')
ax.set_xticks(index+2*0.4)
ax.set_xticklabels(['1','2','3','4','5'])
# ax.set_yticks(np.linspace(0,1,11))
ax.set_xlabel('No. of 5-Fold Cross Validation')
ax.set_ylabel('Accuracy')

# ax.set_title(r'不同编码方式效果比较', FontProperties=font)

plt.show()

#%% compare different ENCODERs(比较不同编码的效果)
import numpy as np
import pandas as pd
#from time import time
import time
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import scipy.io as sio
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import scale, StandardScaler
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from dimensionality_reduction_com import pca, kernelPCA, factorAnalysis
import keras.utils as utils
from keras.metrics import *
import pickle

start = time.time()
Sepscores=[]
for name in ['data_EWA.mat','PAAC_Yeast_11.mat','data_yeast_Auto_11.mat']:
    #生成输入数据
    data_train = sio.loadmat(r'D:\Python\My__python\蛋白质相互作用\Ensemble_methods\%s'%name)
    name1=list(data_train.keys())[-1]
    data=data_train.get(name1)
    row=data.shape[0]
    column=data.shape[1]
    index = [i for i in range(row)]
    np.random.shuffle(index)
    index=np.array(index)
    shu=data[:,np.array(range(1,column))]
    #shu=scale(shu)
    label=data[:,0]
    
    #数据降维
    data_2=factorAnalysis(shu,percentage=0.95)#The rate of contribution
    shu=data_2
    
    X=shu[index,:]
    y=label[index]
    
    #模型初始参数设置
    params1 = {'kernel': 'rbf', 'probability':True}
    model1 = SVC(**params1)#using support vector machine
        
    skf= StratifiedKFold(n_splits=5)
    
    j=0
    sepscores1=[]
    for train, test in skf.split(X,y):
        print('5折交叉验证：第%d次'%j)
        x_train=X[train]
        y_train=y[train]
        x_test=X[test]
        y_test=y[test]
        #模型训练（训练集）
        model1=model1.fit(x_train, y_train)
        #模型验证（测试集）
        y_score1=model1.predict_proba(x_test)
        #模型性能评价
        y_test2=utils.to_categorical(y_test)
        fpr1, tpr1, _ = roc_curve(y_test2[:,0], y_score1[:,0])
        p1, r1, _ = precision_recall_curve(y_test2[:,0], y_score1[:,0])
        roc_auc1 = auc(fpr1, tpr1)
        y_class1 = np.argmax(y_score1,axis=1)
        m = Accuracy()
        m.update_state(y_test, y_class1); acc1 = m.result().numpy(); m.reset_states()
        m = Precision()
        m.update_state(y_test, y_class1); precision1 = m.result().numpy(); m.reset_states()
        m = Recall()
        m.update_state(y_test, y_class1); recall1 = m.result().numpy(); m.reset_states()
        j+=1
        sepscores1.append([acc1,precision1,recall1,roc_auc1])
    Sepscores.append(sepscores1)

Sepscores=np.array(Sepscores)
#画图
font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)
index=np.array([1,5,9,13,17])
fig=plt.figure(figsize=[11, 8])
ax=fig.add_subplot(1,1,1)
ax.bar(index, Sepscores[0,:,0], width=1.0, label='IT-encoder')
ax.bar(index+1, Sepscores[1,:,0], width=1.0, label='PseAAC')
ax.bar(index+2, Sepscores[2,:,0], width=1.0, label='AC')
ax.legend(loc='lower right')
ax.set_xticks(index+1)
ax.set_xticklabels(['1','2','3','4','5'])
ax.set_yticks(np.linspace(0,1,11))
ax.set_xlabel('No. of 5-Fold Cross Validation')
ax.set_ylabel('Accuracy')

# ax.set_title(r'不同编码方式效果比较', FontProperties=font)

plt.show()

#%% compare different MODELS(比较不同模型的效果)
import numpy as np
import pandas as pd
#from time import time
import time
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import scipy.io as sio
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import scale, StandardScaler
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from dimensionality_reduction_com import pca, kernelPCA, factorAnalysis
import keras.utils as utils
from keras.metrics import *
import pickle

start = time.time()
Sepscores=[]
for name in ['data_EWA.mat','PAAC_Yeast_11.mat','data_yeast_Auto_11.mat']:
    #生成输入数据
    data_train = sio.loadmat(r'D:\Python\My__python\蛋白质相互作用\Ensemble_methods\%s'%name)
    name1=list(data_train.keys())[-1]
    data=data_train.get(name1)
    row=data.shape[0]
    column=data.shape[1]
    index = [i for i in range(row)]
    np.random.shuffle(index)
    index=np.array(index)
    shu=data[:,np.array(range(1,column))]
    #shu=scale(shu)
    label=data[:,0]
    
    #数据降维
    data_2=factorAnalysis(shu,percentage=0.95)#The rate of contribution
    shu=data_2
    
    X=shu[index,:]
    y=label[index]
    
    #模型初始参数设置
    params1 = {'kernel': 'rbf', 'probability':True}
    params2 = {'loss': 'deviance', 'learning_rate': 0.1,
              'n_estimators': 1000, 'subsample': 1.0, 'criterion':'friedman_mse', 
              'min_samples_split': 2, 'min_samples_leaf': 1, 'min_weight_fraction_leaf': 0.0, 
              'max_depth': 70,'min_impurity_decrease': 0.0, 'min_impurity_split': None, 
              'max_features': 'sqrt', 'verbose': 0, 'max_leaf_nodes': None, 
              'warm_start': False, 'presort': 'auto'} #the parameters of GTB
    params5 = {'in_dim':X.shape[1], 'n_hidden_1':512, 'n_hidden_2':128, 'out_dim':2}
    model1 = SVC(**params1)#using support vector machine
    model2 = GradientBoostingClassifier(**params2)
    # model3 = CNNet()
    # model4 = CNNet1()
    # model5 = BPNNet(**params5)
    model6 = KNeighborsClassifier(n_neighbors=10,weights="distance",algorithm="auto")
    model8 = RandomForestClassifier(n_estimators=1000, criterion='gini', max_depth=70, 
                                    min_samples_split=2, min_samples_leaf=1, 
                                    min_weight_fraction_leaf=0.0, max_features='auto', 
                                    max_leaf_nodes=None, min_impurity_decrease=0.0, 
                                    min_impurity_split=None, bootstrap=True, 
                                    oob_score=False, n_jobs=1, random_state=None, verbose=0, 
                                    warm_start=False, class_weight=None)#the parameters of RF
    
    skf= StratifiedKFold(n_splits=5)
    
    j=0
    sepscores1=[]
    for train, test in skf.split(X,y):
        print('5折交叉验证：第%d次'%j)
        x_train=X[train]
        y_train=y[train]
        x_test=X[test]
        y_test=y[test]
        #模型训练（训练集）
        model1=model1.fit(x_train, y_train)
        model2=model2.fit(x_train, y_train)
        model6=model6.fit(x_train, y_train)
        model8=model8.fit(x_train, y_train)
        #模型验证（测试集）
        y_score1=model1.predict_proba(x_test)
        y_score2=model2.predict_proba(x_test)
        # y_score3=model3.predict_proba(x_test)
        # y_score4=model4.predict_proba(x_test)
        # y_score5=Model[-1].predict_proba(x_test)
        y_score6=model6.predict_proba(x_test)
        # y_score7=model7(X[train],y[train],X[test])
        y_score8=model8.predict_proba(x_test)
        #模型性能评价
        y_test2=utils.to_categorical(y_test)
        fpr1, tpr1, _ = roc_curve(y_test2[:,0], y_score1[:,0])
        p1, r1, _ = precision_recall_curve(y_test2[:,0], y_score1[:,0])
        fpr2, tpr2, _ = roc_curve(y_test2[:,0], y_score2[:,0])
        p2, r2, _ = precision_recall_curve(y_test2[:,0], y_score2[:,0])
        # fpr3, tpr3, _ = roc_curve(y_test2[:,0], y_score3[:,0])
        # p3, r3, _ = precision_recall_curve(y_test2[:,0], y_score3[:,0])
        # fpr4, tpr4, _ = roc_curve(y_test2[:,0], y_score4[:,0])
        # p4, r4, _ = precision_recall_curve(y_test2[:,0], y_score4[:,0])
        # fpr5, tpr5, _ = roc_curve(y_test2[:,0], y_score5[:,0])
        # p5, r5, _ = precision_recall_curve(y_test2[:,0], y_score5[:,0])
        fpr6, tpr6, _ = roc_curve(y_test2[:,0], y_score6[:,0])
        p6, r6, _ = precision_recall_curve(y_test2[:,0], y_score6[:,0])
        fpr8, tpr8, _ = roc_curve(y_test2[:,0], y_score8[:,0])
        p8, r8, _ = precision_recall_curve(y_test2[:,0], y_score8[:,0])
        # fpr, tpr, _ = roc_curve(y_test2[:,0], y_score[:,0])
        # p, r, _ = precision_recall_curve(y_test2[:,0], y_score[:,0])
        roc_auc1 = auc(fpr1, tpr1)
        roc_auc2 = auc(fpr2, tpr2)
        # roc_auc3 = auc(fpr3, tpr3)
        # roc_auc4 = auc(fpr4, tpr4)
        # roc_auc5 = auc(fpr5, tpr5)
        roc_auc6 = auc(fpr6, tpr6)
        roc_auc8 = auc(fpr8, tpr8)
        # roc_auc = auc(fpr, tpr)
        y_class1 = np.argmax(y_score1,axis=1)
        y_class2 = np.argmax(y_score2,axis=1)
        # y_class3 = np.argmax(y_score3,axis=1)
        # y_class4 = np.argmax(y_score4,axis=1)
        # y_class5 = np.argmax(y_score5,axis=1)
        y_class6 = np.argmax(y_score6,axis=1)
        y_class8 = np.argmax(y_score8,axis=1)
        # y_class = np.argmax(y_score,axis=1)
        
        m = Accuracy()
        m.update_state(y_test, y_class1); acc1 = m.result().numpy(); m.reset_states()
        m.update_state(y_test, y_class2); acc2 = m.result().numpy(); m.reset_states()
        # m.update_state(y_test, y_class3); acc3 = m.result().numpy(); m.reset_states()
        # m.update_state(y_test, y_class4); acc4 = m.result().numpy(); m.reset_states()
        # m.update_state(y_test, y_class5); acc5 = m.result().numpy(); m.reset_states()
        m.update_state(y_test, y_class6); acc6 = m.result().numpy(); m.reset_states()
        m.update_state(y_test, y_class8); acc8 = m.result().numpy(); m.reset_states()
        # m.update_state(y_test, y_class); acc = m.result().numpy(); m.reset_states()
        
        m = Precision()
        m.update_state(y_test, y_class1); precision1 = m.result().numpy(); m.reset_states()
        m.update_state(y_test, y_class2); precision2 = m.result().numpy(); m.reset_states()
        # m.update_state(y_test, y_class3); precision3 = m.result().numpy(); m.reset_states()
        # m.update_state(y_test, y_class4); precision4 = m.result().numpy(); m.reset_states()
        # m.update_state(y_test, y_class5); precision5 = m.result().numpy(); m.reset_states()
        m.update_state(y_test, y_class6); precision6 = m.result().numpy(); m.reset_states()
        m.update_state(y_test, y_class8); precision8 = m.result().numpy(); m.reset_states()
        # m.update_state(y_test, y_class); precision = m.result().numpy(); m.reset_states()
        
        m = Recall()
        m.update_state(y_test, y_class1); recall1 = m.result().numpy(); m.reset_states()
        m.update_state(y_test, y_class2); recall2 = m.result().numpy(); m.reset_states()
        # m.update_state(y_test, y_class3); recall3 = m.result().numpy(); m.reset_states()
        # m.update_state(y_test, y_class4); recall4 = m.result().numpy(); m.reset_states()
        # m.update_state(y_test, y_class5); recall5 = m.result().numpy(); m.reset_states()
        m.update_state(y_test, y_class6); recall6 = m.result().numpy(); m.reset_states()
        m.update_state(y_test, y_class8); recall8 = m.result().numpy(); m.reset_states()
        # m.update_state(y_test, y_class); recall = m.result().numpy(); m.reset_states()
    
    
        sepscores1.append([[acc1,precision1,recall1,roc_auc1],\
                          [acc2,precision2,recall2,roc_auc2],\
                              # [acc5,precision5,recall5,roc_auc5],\
                                  [acc6,precision6,recall6,roc_auc6],\
                                      [acc8,precision8,recall8,roc_auc8]])
                                          # [acc,precision,recall,roc_auc]])
        j+=1
    Sepscores.append(sepscores1)

Sepscores=np.array(Sepscores)
#画图
font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)
index=np.array([1,5,9,13,17])
fig=plt.figure(figsize=[11, 8])
ax=fig.add_subplot(1,1,1)
#Sepscores第一维编码方法，第二维5-折交叉次序，第三维分类器，第四维性能指标
ax.bar(index, Sepscores[0,:,0,0], width=0.4, label='IT+SVM')
ax.bar(index+1*0.4, Sepscores[1,:,1,0], width=0.4, label='PseAAC+GBT')
ax.bar(index+2*0.4, Sepscores[0,:,2,0], width=0.4, label='IT+KNN')
ax.bar(index+3*0.4, Sepscores[2,:,2,0], width=0.4, label='AC+KNN')
ax.bar(index+4*0.4, Sepscores[2,:,3,0], width=0.4, label='AC+RF')
ax.bar(index+5*0.4, Sepscores[1,:,3,0], width=0.4, label='PseAAC+RF')
#ax.bar(index+6*0.4, sepscores[:,5,0], width=0.4, label='IT-EnsBPNN')
ax.legend(loc='lower right')
ax.set_xticks(index+3*0.4)
ax.set_xticklabels(['1','2','3','4','5'])
# ax.set_yticks(np.linspace(0,1,11))
ax.set_xlabel('No. of 5-Fold Cross Validation')
ax.set_ylabel('Accuracy')

# ax.set_title(r'不同编码方式效果比较', FontProperties=font)

plt.show()