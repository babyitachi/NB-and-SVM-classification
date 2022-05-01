# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from cvxopt import matrix,solvers
import libsvm.svmutil as svm 
import scipy
import sys

class Args:
    path_of_train_data=""
    path_of_test_data=""
    part_num=""
    def __init__(self, train, test, part):
        self.path_of_train_data = train
        self.path_of_test_data= test
        self.part_num=part
        
#################### Console Arguments ##################
def read_cli():
    a=sys.argv[1]
    b=sys.argv[2]
    c=sys.argv[3]
    args= Args(a,b,c)

    return args

################### functions ###############
def getInputData(trainPath,testPath):
    traindata=pd.read_csv(trainPath)
    testdata=pd.read_csv(testPath)
    return traindata,testdata

def formatData(traindata,testdata):
    # my entry no is 2021SIY7558
    d=8
    d1=int(np.mod(d+1,10))
    classes=[8,9]
    mapy={8:-1,9:1}
    
    testdata=testdata.loc[(testdata['7'] == d) | (testdata['7'] == d1)]
    testdata=np.array(testdata)
    ytest=testdata[:,-1]
    for index,i in enumerate(ytest):
        ytest[index]=mapy.get(i)
    ytest=ytest.reshape(-1,1)
    xtest=testdata[:,:-1].astype('int')
    xtest=np.divide(xtest,255)
    del testdata
    
    traindata=traindata.loc[(traindata['7'] == d) | (traindata['7'] == d1)]
    traindata=np.array(traindata)
    ytrain=traindata[:,-1]
    for index,i in enumerate(ytrain):
        ytrain[index]=mapy.get(i)
    ytrain=ytrain.reshape(-1,1)
    xtrain=traindata[:,:-1].astype('int')
    xtrain=np.divide(xtrain,255)
    del traindata
    return xtrain,ytrain,xtest,ytest,classes

def getGaussianKernalMatrix(data,gamma):
    s=np.shape(data)[0]
    K=np.zeros([s,s])
    for i in range(s):
        for j in range(s):
            K[i][j]=gaussiankernal(data[i,:],data[j,:],gamma)
    return K

def gaussiankernal(x,z,gamma):
    return np.exp(-1*gamma*(np.square(np.linalg.norm(x-z))))

def predict(X,w,b):
        return np.sign(np.dot(X,w)+b)
    
def confusionMatrix(classes,gold,pred):
    cm=np.zeros([len(classes),len(classes)])
    for index,i in enumerate(gold):
        if i==pred[index]:
            cm[int(-(i-1)/2)][int(-(i-1)/2)]=cm[int(-(i-1)/2)][int(-(i-1)/2)]+1
        else:
            cm[int(-(i-1)/2)][int(-(pred[index]-1)/2)]=cm[int(-(i-1)/2)][int(-(pred[index]-1)/2)]+1
    return cm
################### (a) #####################
def parta(xtrain,ytrain,xtest,ytest):
    C=1.0
    m=np.shape(xtrain)[0]
    
    y_X=ytrain*xtrain
    H=np.dot(y_X,y_X.T)*1 ## kernal trick for linear kernal
    P = matrix(H)
    q = matrix(np.ones((m,1)) * -1)
    A = matrix(ytrain, (1,m),'d')
    b = matrix(0.0)
    
    tmp1 = np.diag(np.ones(m) * -1)
    tmp2 = np.identity(m)
    G = matrix(np.vstack((tmp1, tmp2)))
    tmp1 = np.zeros(m)
    tmp2 = np.ones(m) * C
    h = matrix(np.hstack((tmp1, tmp2)))
    
    solution = solvers.qp(P, q, G, h, A, b)
    
    alpha = np.ravel(solution['x'])
    
    threshold=1e-5
    supportvectors = alpha > threshold
    SValpha = alpha[supportvectors]
    SVx = xtrain[supportvectors]
    SVy = ytrain[supportvectors]
    
    w = np.zeros(np.shape(xtrain)[1])
    for n in range(len(SValpha)):
        w = w + (SValpha[n] * SVy[n][0] * SVx[n].reshape(1,-1))
    w=w.T
    
    pos=ytrain==1
    neg=ytrain==-1
    b=-1*(max(np.dot(xtrain,w)[neg])+min(np.dot(xtrain,w)[pos]))/2
    
    pred=predict(xtest,w,b)
    
    acc_bc=np.sum(np.equal(ytest,pred))/len(ytest)
    return acc_bc,w,b,pred

################ (b) ########################
def partb(xtrain,ytrain,xtest,ytest):
    gamma=0.05 
    C=1.0
    m=np.shape(xtrain)[0]
    gaussianKernalMatrix=getGaussianKernalMatrix(xtrain,gamma)
    #gaussianKernalMatrix = np.load('GK.npy')
    
    P = matrix(np.outer(ytrain,ytrain) * gaussianKernalMatrix)
    q = matrix(np.ones((m,1)) * -1)
    A = matrix(ytrain, (1,m),'d')
    b = matrix(np.zeros([1,1]))
    
    tmp1 = np.diag(np.ones(m) * -1)
    tmp2 = np.identity(m)
    G = matrix(np.vstack((tmp1, tmp2)))
    tmp1 = np.zeros(m)
    tmp2 = np.ones(m) * C
    h = matrix(np.hstack((tmp1, tmp2)))
    
    solution = solvers.qp(P, q, G, h, A, b)
    
    alpha_gau = np.ravel(solution['x'])
    
    threshold=1e-5
    supportvectors_gau = alpha_gau > threshold
    SValpha_gau = alpha_gau[supportvectors_gau]
    SVx_gau = xtrain[supportvectors_gau]
    SVy_gau = ytrain[supportvectors_gau]
#    print("There are",len(SValpha_gau),"support vectors from",m,"points, for Gaussian kernal")
    
    w_gau = np.zeros(np.shape(xtrain)[1])
    for n in range(len(SValpha_gau)):
        w_gau = w_gau + (SValpha_gau[n] * SVy_gau[n][0] * SVx_gau[n].reshape(1,-1))
    w_gau=w_gau.T
    
    pos=ytrain==1
    neg=ytrain==-1
    b_gau=-1*(max(np.dot(xtrain,w_gau)[neg])+min(np.dot(xtrain,w_gau)[pos]))/2
    
    pred_gau=predict(xtest,w_gau,b_gau)
    
    acc_gau_bc=np.sum(np.equal(ytest,pred_gau))/len(ytest)
    
    return acc_gau_bc,w_gau,b,pred_gau

#################### (c) ##########################

def partc(xtrain,ytrain,xtest,ytest):
    y=scipy.asarray(ytrain.flatten())
    x=scipy.sparse.csr_matrix(xtrain)
    
    y_test=scipy.asarray(ytest.flatten())
    x_test=scipy.sparse.csr_matrix(xtest)
    
    prob  = svm.svm_problem(y, x, isKernel=True)
    param_li = svm.svm_parameter('-t 0 -c 1')
    m_li = svm.svm_train(prob, param_li)
    p_label_li, p_acc_li, p_val_li=svm.svm_predict(y_test,x_test,m_li)
    SV_lin=m_li.get_SV()
    noofSV_lin=len(SV_lin)
    
    param_gau = svm.svm_parameter('-t 2 -g 0.05 -c 1.0')
    m_gau = svm.svm_train(prob, param_gau)
    p_label_gau, p_acc_gau, p_val_gau=svm.svm_predict(y_test,x_test,m_gau)
    SV_gau=m_gau.get_SV()
    noofSV_gau=len(SV_gau)

#    cm=confusionMatrix(classes,ytest,pred)
    return p_acc_li,p_acc_gau,noofSV_lin,noofSV_gau

#################### Main Func ############################
if __name__ == '__main__':
    args = read_cli()
    traindata,testdata = getInputData(args.path_of_train_data,args.path_of_test_data)
    xtrain,ytrain,xtest,ytest,classes=formatData(traindata,testdata)
    if args.part_num=='a':
        acc_a,w_a,b_a,pred_a=parta(xtrain,ytrain,xtest,ytest)
        print('accuracy',acc_a)
        print('bais',b_a)
    elif args.part_num=='b':
        acc_b,w_b,b_b,pred_b=partb(xtrain,ytrain,xtest,ytest)
        print('accuracy',acc_b)
        print('bias',b_b)
    elif args.part_num=='c':
        acc_li,acc_gau,noLisv,noGasv=partc(xtrain,ytrain,xtest,ytest)
        print('linear accuracy',acc_li[0],',','no. of linear Support Vectors',noLisv)
        print('gaussian accuracy',acc_gau[0],',','no. of Gaussian Support Vectors',noGasv)
    else:
        print('Invald part selection.')