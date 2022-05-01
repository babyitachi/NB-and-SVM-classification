# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from cvxopt import matrix,solvers
import libsvm.svmutil as svm 
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
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

def confusionMatrix(classes,gold,pred):
    cm=np.zeros([len(classes),len(classes)])
    for index,i in enumerate(gold):
        if i==pred[index]:
            cm[i][i]=cm[i][i]+1
        else:
            cm[i][int(pred[index])]=cm[i][int(pred[index])]+1
    return cm

################### functions ###############
def getInputData(trainPath,testPath):
    traindata=pd.read_csv(trainPath)
    testdata=pd.read_csv(testPath)
    return traindata,testdata


def formatData(traindata,testdata):
    classes=[0,1,2,3,4,5,6,7,8,9]
    
    testdata=np.array(testdata)
    ytest=testdata[:,-1]
    ytest=ytest.reshape(-1,1)
    xtest=testdata[:,:-1].astype('int')
    xtest=np.divide(xtest,255)
    del testdata
    
    traindata=np.array(traindata)
    ytrain=traindata[:,-1]
    ytrain=ytrain.reshape(-1,1)
    xtrain=traindata[:,:-1].astype('int')
    xtrain=np.divide(xtrain,255)
    del traindata
    return xtrain,ytrain,xtest,ytest,classes

#   here we have k=10
# so kC2 = 10C2 = 45
# we have to create 45 different classifiers for each pair of numbers

################### (a) #####################
def parta():
    print("Not Implemented")






################### (b) #####################
def partb(xtrain,ytrain,xtest,ytest):
    y=scipy.asarray(ytrain.flatten())
    x=scipy.sparse.csr_matrix(xtrain)
    
    y_test=scipy.asarray(ytest.flatten())
    x_test=scipy.sparse.csr_matrix(xtest)
    
    prob  = svm.svm_problem(y, x, isKernel=True)
    param_gau = svm.svm_parameter('-t 2 -g 0.05 -c 1.0')
    m_gau = svm.svm_train(prob, param_gau)
    p_label_gau, p_acc_gau, p_val_gau=svm.svm_predict(y_test,x_test,m_gau)
#    SV_gau=m_gau.get_SV()
    
    return p_acc_gau,p_label_gau

# the LIBSVM takes lesser time for computation 
# while on part a, for calculation gaussian kernal it takes significant amount of time

###################### (c) ####################

def partc(classes,gold,pred):
    
    cm=confusionMatrix(classes,gold.flatten(),pred)
    ax= plt.subplot()
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['Class 0','Class 1','Class 2','Class 3','Class 4', 'Class 5','Class 6','Class 7','Class 8','Class 9'])
    ax.yaxis.set_ticklabels(['Class 0','Class 1','Class 2','Class 3','Class 4', 'Class 5','Class 6','Class 7','Class 8','Class 9'])
    plt.figure(figsize = (10,10))
    sns.heatmap(cm, annot=True,ax=ax)
    return cm

# class 1 are the most miss-classified as the class 7, which is 48 miss classifications
# '2' is miss-classified as '0' 13 time and as a '8' also 13 times
# '9' is miss-classified as '4' 36, as '5' 24 and as '8' 10 times
# '5' is miss-classified as '3' 32, as '6' 8 and as '8' 7 times
# '8' is miss-classified as '3' 11, as '5' 10 and as '2' 7 times

# yes, result does make sense

######################## (d) ####################
def partd(xtrain,ytrain,xtest,ytest):
    Cvals=[1e-5,1e-3,1,5,10]
    
    y=scipy.asarray(ytrain.flatten())
    x=scipy.sparse.csr_matrix(xtrain)
    
    y_test=scipy.asarray(ytest.flatten())
    x_test=scipy.sparse.csr_matrix(xtest)
    
    valmodels=[]
    accuracies=[]
    labeles=[]
    models=[]
    accuracies_test=[]
    
    
    for i in Cvals:
        prob  = svm.svm_problem(y, x)
        param_gau_val = svm.svm_parameter('-t 2 -v 5 -g 0.05 -c '+str(i))
        m_gau = svm.svm_train(prob, param_gau_val)
        accuracies.append(m_gau)
    
    for i in Cvals:
        prob  = svm.svm_problem(y, x)
        param_gau_val = svm.svm_parameter('-t 2 -g 0.05 -c '+str(i))
        m_gau = svm.svm_train(prob, param_gau_val)
        p_label_gau, p_acc_gau, p_val_gau = svm.svm_predict(y_test,x_test,m_gau)
        valmodels.append(p_val_gau)
        accuracies_test.append(p_acc_gau)
        labeles.append(p_label_gau)
        models.append(m_gau)
    return accuracies,accuracies_test
            
    

#################### Main Func ############################
if __name__ == '__main__':
    args = read_cli()
    traindata,testdata = getInputData(args.path_of_train_data,args.path_of_test_data)
    xtrain,ytrain,xtest,ytest,classes=formatData(traindata,testdata)
    if args.part_num=='a':
        parta()
    elif args.part_num=='b':
        acc_b,pred_b=partb(xtrain,ytrain,xtest,ytest)
        np.save('./pred_b',pred_b)
        print('accuracy',acc_b[0])
    elif args.part_num=='c':
        pred_b=np.load('./pred_b.npy')
        cm_c=partc(classes,ytest,pred_b)
        print('CM: ',cm_c)
        np.save('./cm_c',cm_c)
    elif args.part_num=='d':
        val_acc,test_acc=partd(xtrain,ytrain,xtest,ytest)
        print('Validation Accuarcy is',val_acc)
        print('Test Accuarcy is',test_acc)
    else:
        print('Invald part selection.')