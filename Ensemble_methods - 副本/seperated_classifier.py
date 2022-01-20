# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 15:41:55 2020

@author: 25326
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torchcontrib
import matplotlib.pyplot as plt
import copy 
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from keras.metrics import *
from sklearn.decomposition import sparse_encode

class CNNet(nn.Module):
    def __init__(self):
        # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        super().__init__()
        self.height = 10
        self.width = 10
        self.conv1 = nn.Conv2d(64, 256, (10,10))
        self.conv2 = nn.Conv2d(256, 64, (1,1))
        self.fc1 = nn.Linear(64, 2)
        self.drop = nn.Dropout2d(p=0.7)

    def __forward(self, x):
        
        x = self.conv1(x)
        x = F.relu(x)
        x = self.drop(x)
        # print(x.size())
        x = self.conv2(x)
        # x = F.max_pool2d(x, (2, 2))
        x = F.relu(x)
        x = self.drop(x)
        # print(x.size())
        # reshape，‘-1’表示自适应
        x = x.view(x.size()[0], -1)
        # print(x.size())
        x = self.fc1(x)
        # x = F.sigmoid(x)
        return x
    
    def fit(self,X,y,X_t,y_t,init=True,n_batches=1,n_epoch=10000):
        if n_batches<=1:#判断是否要分批训练
            Batch=[(list(range(X.shape[0])),None)]
        else:
            Batch=StratifiedKFold(n_splits=n_batches).split(X,y)
        if init==True:
            self.__init__()
                
        print('CNN训练开始：', time.strftime('%Y-%m-%d %H:%M:%S')+'\n')
        Loss = []
        Score = []
        i=1
        for train,_ in Batch:
            x_train=X[train]#可以更换为test使其为不重复的分批训练
            y_train=y[train]
            # if x_train.shape[1]<self.height*self.width:
            #     x_train = np.column_stack((x_train,np.zeros((x_train.shape[0],self.height*self.width-x_train.shape[1]))))
            #     x_train = torch.tensor(x_train).float().reshape(x_train.shape[0],1,self.height,self.width)
            # else:
            #     x_train = torch.tensor(x_train[:,0:self.height*self.width]).float().reshape(x_train.shape[0],1,self.height,self.width)
            x_train = torch.tensor(x_train).float()
            y_train = torch.tensor(y_train).long()
            X_t = torch.tensor(X_t).float()
            y_t = torch.tensor(y_t).long()
            # 新建一个优化器，只需要要调整的参数和学习率
            optimizer = torch.optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
            #optimizer = torch.optim.SGD(self.parameters(), lr=0.05)
            criterion = nn.CrossEntropyLoss()
            
            
            for epoch in range(n_epoch):
                out = self.__forward(x_train)
                loss = criterion(out, y_train)
                # loss是个scalar，我们可以直接用item获取到他的python类型的数值
                Loss.append(loss.item())
                if epoch%20 == 0:
                    print('批次：%d'%i,'次数：%d'%epoch,'损失：%f'%loss.item())
                    self.eval()
                    result1 = self.__forward(x_train).data
                    result2 = self.__forward(X_t).data
                    self.train()
                    y_class1 = np.argmax(result1,axis=1)
                    y_class2 = np.argmax(result2,axis=1)
                    m = Accuracy()
                    m.update_state(y_train, y_class1); score1 = m.result().numpy(); m.reset_states()
                    m.update_state(y_t, y_class2); score2 = m.result().numpy(); m.reset_states()
                    Score.append(score2)
                    print('训练集预测正确率：', score1)
                    print('测试集预测正确率：', score2)
                    print(time.strftime('%Y-%m-%d %H:%M:%S')+'\n')
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            i+=1
        print('CNN训练结束：', time.strftime('%Y-%m-%d %H:%M:%S')+'\n')
        plt.plot(Loss)
        return self
    
    def predict_proba(self,x_test):
        # if x_test.shape[1]<self.height*self.width:
        #     x_test = np.column_stack((x_test,np.zeros((x_test.shape[0],self.height*self.width-x_test.shape[1]))))
        #     x_test = torch.tensor(x_test).float().reshape(x_test.shape[0],1,self.height,self.width)
        # else:
        #     x_test = torch.tensor(x_test[:,0:self.height*self.width]).float().reshape(x_test.shape[0],1,self.height,self.width)
        x_test = torch.tensor(x_test).float()
        self.eval()
        result=self.__forward(x_test)
        return np.array(torch.softmax(result,dim=1).data)
    
    
class CNNet1(nn.Module):
    def __init__(self):
        # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        super().__init__()
        self.height = 8
        self.width = 8
        self.conv1 = nn.Conv2d(1, 256, (5,5))
        self.conv2 = nn.Conv2d(256, 64, (4,4))
        self.fc1 = nn.Linear(64, 2)
        self.drop = nn.Dropout2d(p=0.7)

    def __forward(self, x):
        
        x = self.conv1(x)
        x = F.relu(x)
        x = self.drop(x)
        # print(x.size())
        x = self.conv2(x)
        # x = F.max_pool2d(x, (2, 2))
        x = F.relu(x)
        x = self.drop(x)
        # print(x.size())
        # reshape，‘-1’表示自适应
        x = x.view(x.size()[0], -1)
        # print(x.size())
        x = self.fc1(x)
        # x = F.sigmoid(x)
        return x
    
    def fit(self,X,y,n_batches=1,init=True,n_epoch=10000):
        if n_batches<=1:#判断是否要分批训练
            Batch=[list(range(X.shape[0]))]
        else:
            Batch=StratifiedKFold(n_splits=n_batches).split(X,y)
        if init==True:
            self.__init__()
                
        print('CNN1训练开始：', time.strftime('%Y-%m-%d %H:%M:%S')+'\n')
        Loss = []
        i=1
        for train,test in Batch:
            x_train=X[train]#可以更换为test使其为不重复的分批训练
            y_train=y[train]
            
            if x_train.shape[1]<self.height*self.width:
                x_train = np.column_stack((x_train,np.zeros((x_train.shape[0],self.height*self.width-x_train.shape[1]))))
                x_train = torch.tensor(x_train).float().reshape(x_train.shape[0],1,self.height,self.width)
            else:
                x_train = torch.tensor(x_train[:,0:self.height*self.width]).float().reshape(x_train.shape[0],1,self.height,self.width)
            
            y_train = torch.tensor(y_train).long()
            # 新建一个优化器，只需要要调整的参数和学习率
            optimizer = torch.optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
            #optimizer = torch.optim.SGD(self.parameters(), lr=0.05)
            criterion = nn.CrossEntropyLoss()
            
            for epoch in range(n_epoch):
                out = self.__forward(x_train)
                loss = criterion(out, y_train)
                # loss是个scalar，我们可以直接用item获取到他的python类型的数值
                Loss.append(loss.item())
                if epoch%20 == 0:
                    print('批次：%d'%i,'次数：%d'%epoch,'损失：%f'%loss.item())
                    self.eval()
                    result = self.__forward(x_train).data
                    self.train()
                    y_class = np.argmax(result,axis=1)
                    m = Accuracy()
                    m.update_state(y_train, y_class); score = m.result().numpy(); m.reset_states()
                    print('训练集预测正确率：', score)
                    print(time.strftime('%Y-%m-%d %H:%M:%S')+'\n')
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            i+=1
        print('CNN1训练结束：', time.strftime('%Y-%m-%d %H:%M:%S')+'\n')
        plt.plot(Loss)
        return self
    
    def predict_proba(self,x_test):
        if x_test.shape[1]<self.height*self.width:
            x_test = np.column_stack((x_test,np.zeros((x_test.shape[0],self.height*self.width-x_test.shape[1]))))
            x_test = torch.tensor(x_test).float().reshape(x_test.shape[0],1,self.height,self.width)
        else:
            x_test = torch.tensor(x_test[:,0:self.height*self.width]).float().reshape(x_test.shape[0],1,self.height,self.width)
        
        self.eval()
        result=self.__forward(x_test)
        return np.array(torch.softmax(result,dim=1).data)

class Auto_Encode(nn.Module):
    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1
        # self.dim2 = dim2
        self.encoder = nn.Sequential(nn.Linear(self.dim0, self.dim1),nn.Tanh(),nn.BatchNorm1d(self.dim1))
        self.decoder = nn.Sequential(nn.Linear(self.dim1, self.dim0),nn.Tanh(),nn.BatchNorm1d(self.dim0))
        self.drop = nn.Dropout(p=0.5)

    def forward(self, x):
        y = self.encoder(x)
        x = self.decoder(y)
        return x,y
    
    def L2Loss(self,alpha):
        l2_loss = torch.tensor(0.0,requires_grad = True)
        for name,param in self.named_parameters():
            if 'bias' not in name:
                l2_loss = l2_loss + (0.5*alpha * torch.sum(torch.pow(param,2)))
        return l2_loss
    
    def Variance(self,beta):
        Var = torch.tensor(0.0,requires_grad = True)
        for name,param in self.named_parameters():
            Var = Var + (0.5*beta * torch.var(param))
        return Var

    def focal_loss_with_regularization(self,y_pred,y_true):#加了正则化的损失函数
        focal = nn.MSELoss()(y_pred,y_true)
        # l2_loss = self.L2Loss(0.01)
        # Var = self.Variance(30)
        total_loss = focal #+ Var
        return total_loss
    
    def fit(self,x_train,init=True,n_epoch=1000):
        if init==True:
            self.__init__(self.dim0, self.dim1)
                
        print('Auto_Encode训练开始：', time.strftime('%Y-%m-%d %H:%M:%S')+'\n')
        Loss=[]
        x_train = torch.tensor(x_train).float()
        y_train = x_train
        # 新建一个优化器，需要调整参数和学习率及正则化衰减系数
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        #optimizer = torch.optim.SGD(self.parameters(), lr=0.05)
        criterion = self.focal_loss_with_regularization
        
        for epoch in range(n_epoch):
            out,code = self.forward(x_train)
            loss = criterion(out, y_train)
            # 用item获取到他的python类型的数值
            Loss.append(loss.item())
            if epoch%50 == 0:
                print('次数：%d'%epoch,'损失：%f'%loss.item())
                print(time.strftime('%Y-%m-%d %H:%M:%S')+'\n')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Auto_Encode训练结束：', time.strftime('%Y-%m-%d %H:%M:%S')+'\n')
        plt.plot(Loss)
        return self
    
    def Encode(self,X):
        X = torch.tensor(X).float()
        X = self.encoder(X)
        return X.detach().numpy()

from torch.optim.lr_scheduler import LambdaLR
class BPNNet(nn.Module):
    def __init__(self,in_dim, n_hidden_1, n_hidden_2, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.n_hidden_1 = n_hidden_1
        self.n_hidden_2 = n_hidden_2
        self.out_dim = out_dim
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1),nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim),nn.Sigmoid())
        self.drop1 = nn.Dropout(p=0.2)
        self.drop2 = nn.Dropout(p=0.7)
        self.drop3 = nn.Dropout(p=0.7)
        self.drop4 = nn.Dropout(p=0.2)
        self.num_param = sum(p.numel() for name,p in self.named_parameters() if (p.requires_grad)&('bias' not in name))

    def forward(self, x):
        x = self.drop1(x)
        x = self.layer1(x)
        x = self.drop2(x)
        x = self.layer2(x)
        x = self.drop3(x)
        x = self.layer3(x)
        x = self.drop4(x)
        return x
    
    def L2Loss(self,alpha):
        l2_loss = torch.tensor(0.0,requires_grad = True)
        for name,param in self.named_parameters():
            if 'bias' not in name:
                l2_loss = l2_loss + (0.5*alpha * torch.sum(torch.pow(param,2)))
        return l2_loss

    def focal_loss_with_regularization(self,y_pred,y_true):#加了正则化的损失函数
        focal = nn.CrossEntropyLoss()(y_pred,y_true)
        l2_loss = self.L2Loss(0.00)
        total_loss = focal + l2_loss
        return total_loss
    
    def rate(self,epoch,r1=0.101,r2=0.1,T=200,T1=1500):
        if epoch<T1:
            return 1
        else:
            t=1/T*((epoch-1-T1)%T+1)
            if t<0.5:
                r=(1-2*t)*r1+2*t*r2
            else:
                r=(2-2*t)*r2+(2*t-1)*r1
            return r
    
    def fit(self,X,y,X_t,y_t,n_batches=1,init=True,n_epoch=10000):
        if n_batches<=1:#判断是否要分批训练
            Batch=[(list(range(X.shape[0])),None)]
        else:
            Batch=StratifiedKFold(n_splits=n_batches).split(X,y)
        if init==True:
                self.__init__(self.in_dim, self.n_hidden_1, self.n_hidden_2, self.out_dim)
                
        print('BPNN训练开始：', time.strftime('%Y-%m-%d %H:%M:%S')+'\n')
        Loss=[];Score=[]
        i=1
        for train,_ in Batch:
            x_train=X[train]#可以更换为test使其为不重复的分批训练
            y_train=y[train]
            
            x_train = torch.tensor(x_train).float()
            y_train = torch.tensor(y_train).long()
            X_t = torch.tensor(X_t).float()
            y_t = torch.tensor(y_t).long()
            Model=[];R=[]
            # 新建一个优化器，需要调整参数和学习率
            base_opt = torch.optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
            optimizer1 = torch.optim.Adam(self.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
            optimizer2 = torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9, nesterov=True)
            optimizer = torchcontrib.optim.SWA(optimizer2)            
            #optimizer = torch.optim.SGD(self.parameters(), lr=0.05)
            scheduler = LambdaLR(optimizer, lr_lambda=self.rate)
            # criterion = nn.CrossEntropyLoss()
            criterion = self.focal_loss_with_regularization
            
            for epoch in range(n_epoch):
                out = self.forward(x_train)
                loss = criterion(out, y_train)
                # 用item获取到他的python类型的数值
                Loss.append(loss.item())
                if epoch%50 == 0:
                    print('批次：%d'%i,'次数：%d'%epoch,'损失：%f'%loss.item())
                    self.eval()
                    result1 = self.forward(x_train).data
                    result2 = self.forward(X_t).data
                    self.train()
                    y_class1 = np.argmax(result1,axis=1)
                    y_class2 = np.argmax(result2,axis=1)
                    m = Accuracy()
                    m.update_state(y_train, y_class1); score1 = m.result().numpy(); m.reset_states()
                    m.update_state(y_t, y_class2); score2 = m.result().numpy(); m.reset_states()
                    Score.append(score2)
                    # if score2>=0.925:
                    #     Model.append(copy.deepcopy(self))
                    print('训练集预测正确率：', score1)
                    print('测试集预测正确率：', score2)
                    print(time.strftime('%Y-%m-%d %H:%M:%S')+'\n')
                r=scheduler.get_lr()
                R.append(r[-1])
                if r[-1]<0.01001:
                    Model.append(copy.deepcopy(self))
                    optimizer.update_swa()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                
            optimizer.swap_swa_sgd()#用平均权重替换原始权重
            i+=1
        print('BPNN训练结束：', time.strftime('%Y-%m-%d %H:%M:%S')+'\n')
        plt.plot(R)
        return copy.deepcopy(self),Loss,Score,Model,R
    
    def predict_proba(self,x_test):
        x_test=torch.tensor(x_test).float()
        self.eval()
        return np.array(self.forward(x_test).data)

from numpy.linalg import norm
from numpy import exp
def WSRC_proba(x_train,y_train,x_test):
    # x_train1=x_train.reshape(x_train.shape[0],x_train.shape[1],1)
    # x_train2=x_train.T.reshape(1,x_train.shape[1],x_train.shape[0])
    # sigmma=np.sum(norm(x_train1-x_train2,axis=1))
    sigmma=0
    for i in range(x_train.shape[0]):
        for j in range(x_train.shape[0]):
            sigmma+=norm(x_train[i].reshape(1,-1)-x_train[j].reshape(1,-1))
    sigmma=sigmma/(x_train.shape[0])**2 #用平均距离作为高斯核宽度sigmma
    y_train=y_train.reshape(1,-1)
    N=x_test.shape[0]
    y_score=np.zeros((N,2))
    for i in range(N):
        e=exp(-norm(x_test[i].reshape(1,-1)-x_train,axis=1)**2/(2*sigmma**2)).reshape(-1,1)
        Y=sparse_encode(X=x_test[i].reshape(1,-1),dictionary=e*x_train,alpha=0.01).reshape(1,-1)
        norm1=norm(np.matmul(Y*y_train,x_train) - x_test[i].reshape(1,-1))
        norm0=norm(np.matmul(Y*(1-y_train),x_train) - x_test[i].reshape(1,-1))
        y_score[i,0]=norm1/(norm0+norm1)
        y_score[i,1]=norm0/(norm0+norm1)
        print(i)
    return y_score
    
    
    

from sklearn.cluster import MiniBatchKMeans
from sklearn import metrics
class RBFNet(nn.Module):
    """
    以高斯核作为径向基函数
    """
    def __init__(self,samples,dim_centure,num_centers):
        """
        :param centers: shape=[center_num,data_dim]
        :param n_out:
        """
        super().__init__()
        self.samples = samples
        self.n_out = 2
        self.num_centers = num_centers # 隐层节点的个数
        self.dim_centure = dim_centure
        self.drop = nn.Dropout(p=0.5) # 设置网络剪枝比例
        kmeans_model = MiniBatchKMeans(n_clusters=self.num_centers,init_size=3*self.num_centers).fit(self.samples)
        labels = kmeans_model.labels_
        silhouette_score = metrics.silhouette_score(self.samples, labels, metric='euclidean')
        print('轮廓系数:',silhouette_score)
        self.centers = nn.Parameter(torch.tensor(kmeans_model.cluster_centers_).float())
        self.beta = nn.Parameter(torch.ones(1, self.num_centers), requires_grad=True)
        # self.beta = torch.ones(1, self.num_centers)*3#设置核宽度
        # self.layer1 = nn.Sequential(nn.Linear(self.dim_centure,self.dim_centure), nn.BatchNorm1d(n_hidden_1))
        # 对线性层的输入节点数目进行了修改
        self.layer2 = nn.Sequential(nn.Linear(self.num_centers+self.dim_centure, 4*(self.num_centers+self.dim_centure), bias=True),nn.ReLU())
        self.layer3 = nn.Sequential(nn.Linear(4*(self.num_centers+self.dim_centure), self.n_out, bias=True), nn.Softmax(dim=1))
        self.__initialize_weights()# 创建对象时自动执行
 
 
    def __kernel_fun(self, batches):
        n_input = batches.size(0)  # number of inputs
        A = self.centers.view(self.num_centers, -1).unsqueeze(0).repeat(n_input, 1, 1)
        B = batches.view(n_input, -1).unsqueeze(1).repeat(1, self.num_centers, 1)
        C = torch.exp(-self.beta.mul((A - B).pow(2).sum(2, keepdim=False)))
        return C
 
    def __forward(self, batches):
        
        x = self.__kernel_fun(batches)
        x = torch.cat([batches, x], dim=1)
        x = self.drop(x)
        x = self.layer2(x)
        x = self.drop(x)
        x = self.layer3(x)
        return x
 
    def __initialize_weights(self, ):
        """
        网络权重初始化
        :return:
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()

    def fit(self,x_train,y_train,init=True,n_epoch=10000):
        if n_batches<=1:#判断是否要分批训练
            Batch=[list(range(X.shape[0]))]
        else:
            Batch=StratifiedKFold(n_splits=n_batches).split(X,y)
        if init==True:
                self.__init__(self.samples,self.num_centers,self.dim_centure)
                
        print('RBFN训练开始：', time.strftime('%Y-%m-%d %H:%M:%S')+'\n')
        Loss = []
        i=1
        for train,test in Batch:
            x_train=X[train]#可以更换为test使其为不重复的分批训练
            y_train=y[train]
            
            x_train = torch.tensor(x_train).float()
            y_train = torch.tensor(y_train).long()
            # 新建一个优化器，只需要要调整的参数和学习率
            optimizer = torch.optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
            #optimizer = torch.optim.SGD(self.parameters(), lr=0.05)
            criterion = nn.CrossEntropyLoss()
            
            for epoch in range(n_epoch):
                out = self.__forward(x_train)
                loss = criterion(out, y_train)
                # loss是个scalar，我们可以直接用item获取到他的python类型的数值
                Loss.append(loss.item())
                if epoch%20 == 0:
                    print('批次：%d'%i,'次数：%d'%epoch,'损失：%f'%loss.item())
                    self.eval()
                    result = self.__forward(x_train).data
                    self.train()
                    y_class = np.argmax(result,axis=1)
                    m = Accuracy()
                    m.update_state(y_train, y_class); score = m.result().numpy(); m.reset_states()
                    print('训练集预测正确率：', score)
                    print(time.strftime('%Y-%m-%d %H:%M:%S')+'\n')
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            i+=1
        print('RBFN训练结束：', time.strftime('%Y-%m-%d %H:%M:%S')+'\n')
        plt.plot(Loss)
        return self
    
    def predict_proba(self,x_test):
        x_test=torch.tensor(x_test).float()
        self.eval()
        result=self.__forward(x_test)
        return np.array(result.data)
 
    def print_network(self):
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print(self)
        print('Total number of parameters: %d' % num_params)
 
class Ensemble(nn.Module):
    def __init__(self,in_dim,out_dim=2):
        super().__init__()
        self.in_dim=in_dim
        self.out_dim=out_dim
        self.layer1=nn.Sequential(nn.Linear(self.in_dim,self.out_dim,bias=True),nn.Softmax(dim=1))
        
    def __forward(self,x):
        x=self.layer1(x)
        return x
    
    def fit(self,x_train,y_train,init=True,n_epoch=1000):
        if init==True:
            self.__init__(self.in_dim,self.out_dim)
        
        x_train = torch.tensor(x_train).float()
        y_train = torch.tensor(y_train).long()
        # 新建一个优化器，只需要要调整的参数和学习率
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        #optimizer = torch.optim.SGD(self.parameters(), lr=0.05)
        criterion = nn.CrossEntropyLoss()
        
        Loss = []
        for epoch in range(n_epoch):
            out = self.__forward(x_train)
            loss = criterion(out, y_train)
            # loss是个scalar，我们可以直接用item获取到他的python类型的数值
            Loss.append(loss.item())
            if epoch%50 == 0:
                print('第%d次循环损失值：'%epoch,loss.item())
                self.eval()
                result = self.__forward(x_train).data
                self.train()
                y_class = np.argmax(result,axis=1)
                m = Accuracy()
                m.update_state(y_train, y_class); score = m.result().numpy(); m.reset_states()
                print('训练集预测正确率：', score)
                print(time.strftime('%Y-%m-%d %H:%M:%S')+'\n')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # plt.plot(Loss)
        return self
    
    def predict_proba(self,x_test):
        x_test=torch.tensor(x_test).float()
        self.eval()
        result=self.__forward(x_test)
        return np.array(result.data)
        





















    
    
    

    
    