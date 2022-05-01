# -*- coding: utf-8 -*-
import json
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
stopwords=stopwords.words('english')
import sys
import os.path

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

################### functions ######################
def getInputData(trainPath,testPath):
    trainingdata=[]
    for line in open(trainPath,'r'):
        trainingdata.append(json.loads(line))
    testingdata=[]
    for line in open(testPath,'r'):
        testingdata.append(json.loads(line))
    # here y=1,2,3,4,5
    classes=[1,2,3,4,5]
    return trainingdata,testingdata,classes

def preprocess(text):
    text=re.sub(r'\s+',' ',text,flags=re.I)
    text=text.lower()
    return text    

def getFilteredWords(data):
    words=[]
    for index,i in enumerate(data):
        words.extend([w for w in i['review'].split(" ")])
    return words

def getDict(words):
    dictionary={}
    for i in words:
        if i!="":
            i=i.lower()
            if i in dictionary.keys():
                dictionary[i]=dictionary[i]+1
            else:
                dictionary[i]=1
    return dictionary

def getWordWordProbByClass(nclass,words,filteredwords,filteredwordsSum):
    sumprob=0
    for i in words:
        if i in filteredwords[nclass].keys():
            sumprob=sumprob + np.log(filteredwords[nclass].get(i)/filteredwordsSum[nclass])
        else:
            return 0
    return sumprob

def predict(reviewtext,probR,filteredwords,filteredwordsSum):
    reviewtext=preprocess(reviewtext)
    reviewtext=reviewtext.split(" ")
    nclasses=5
    probClass=[]
    for i in range(nclasses):
        probClass.append(np.exp(getWordWordProbByClass(i,reviewtext,filteredwords,filteredwordsSum)+np.log(probR[i]/sum(probR)))+1)
    m=max(probClass)
    return probClass.index(m)+1

def confusionMatrix(classes,gold,pred):
    cm=np.zeros([len(classes),len(classes)])
    for index,i in enumerate(gold):
        if i==pred[index]:
            cm[i-1][i-1]=cm[i-1][i-1]+1
        else:
            cm[i-1][pred[index]-1]=cm[i-1][pred[index]-1]+1
    return cm
def preprocesswithstopwords(text):
    lemmatizer = WordNetLemmatizer()
    text=text.lower()
    text=re.sub('[^a-zA-Z]+',' ',text)
    text=re.sub(r'\s+[a-zA-Z]\s+',' ',text)
    text=re.sub('^[a-zA-Z]\s+',' ',text)
    text=re.sub(r'\s+',' ',text,flags=re.I)
    splittext=text.split(" ")
    nonstoptext=[]
    for i in splittext:
        if i.strip() not in stopwords:
            nonstoptext.append(lemmatizer.lemmatize(i.strip()))
    text=' '.join(nonstoptext) 
    return text
def predictwithoutstopwords(reviewtext,probR,filteredwords,filteredwordsSum):
    reviewtext=preprocesswithstopwords(reviewtext.strip())
    reviewtext=reviewtext.split(" ")
    nclasses=5
    probClass=[]
    for i in range(nclasses):
        probClass.append(np.exp(getWordWordProbByClass(i,reviewtext,filteredwords,filteredwordsSum)+np.log(probR[i]/sum(probR)))+1)
    m=max(probClass)
    return probClass.index(m)+1

def sumgetFilteredWords(data,field='review'):
    words=[]
    for index,i in enumerate(data):
        words.extend([w for w in i[field].split(" ")])
    return words
def sumgetWordWordProbByClass(nclass,words,sumfilteredwords,sumfilteredwordsSum):
    sumprob=0
    for i in words:
        if i in sumfilteredwords[nclass].keys():
            sumprob=sumprob + np.log(sumfilteredwords[nclass].get(i)/sumfilteredwordsSum[nclass])
        else:
            return 0
    return sumprob
def sumpredict(reviewtext,summary,probR,filteredwords,filteredwordsSum,sumfilteredwords,sumfilteredwordsSum):
    reviewtext=preprocess(reviewtext)
    reviewtext=reviewtext.split(" ")
    summary=preprocess(summary)
    summary=summary.split(" ")
    nclasses=5
    probClass=[]
    for i in range(nclasses):
        l=np.exp(getWordWordProbByClass(i,reviewtext,filteredwords,filteredwordsSum)+np.log(probR[i]/sum(probR)))
        t=np.exp(sumgetWordWordProbByClass(i,summary,sumfilteredwords,sumfilteredwordsSum)+np.log(probR[i]/sum(probR)))
        probClass.append(l+t+1)
    m=max(probClass)
    return probClass.index(m)+1
def getFilteredBigrams(data):
        words=[]
        for index,i in enumerate(data):
            i=list(i['review'].split(" "))
            for first, second in zip(i,i[1:]):
                words.append(first+" "+second)
        return words
def getWordWordProbByClassBigram(nclass,words,filteredwordsBigram,filteredwordsBigramSum):
        sumprob=0
        birds=[]
        for first, second in zip(words,words[1:]):
                birds.append(first+" "+second)
        for i in birds:
            if i in filteredwordsBigram[nclass].keys():
                sumprob=sumprob + np.log(filteredwordsBigram[nclass].get(i)/filteredwordsBigramSum[nclass])
            else:
                return 0
        return sumprob
def predictwithoutstopwordsBigram(reviewtext,filteredwordsBigram,filteredwordsBigramSum,probR):
        reviewtext=preprocesswithstopwords(reviewtext.strip())
        nclasses=5
        probClass=[]
        for i in range(nclasses):
            probClass.append(np.exp(getWordWordProbByClassBigram(i,reviewtext,filteredwordsBigram,filteredwordsBigramSum)+np.log(probR[i]/sum(probR)))+1)
        m=max(probClass)
        print(probClass)
        return probClass.index(m)+1
    
    
################## (a) #############################
def parta(trainingdata,testingdata):
    review=[]
    for i in trainingdata:
        review.append({'rating':i['overall'],'review':i['reviewText']})
    trainingdata=review
    del review 

    for i in trainingdata:
        i['review']=preprocess(i['review'])
    
    trainingclass1=[]
    trainingclass2=[]
    trainingclass3=[]
    trainingclass4=[]
    trainingclass5=[]
    
    probR=[0,0,0,0,0]
    for i in trainingdata:
        if i['rating']==1.0:
            probR[0]=probR[1]+1
            trainingclass1.append(i)
        elif i['rating']==2.0:
            probR[1]=probR[1]+1
            trainingclass2.append(i)
        elif i['rating']==3.0:
            probR[2]=probR[2]+1
            trainingclass3.append(i)
        elif i['rating']==4.0:
            probR[3]=probR[3]+1
            trainingclass4.append(i)
        else:
            probR[4]=probR[4]+1
            trainingclass5.append(i)

    filteredwords1=getDict(getFilteredWords(trainingclass1))
    filteredwords2=getDict(getFilteredWords(trainingclass2))
    filteredwords3=getDict(getFilteredWords(trainingclass3))
    filteredwords4=getDict(getFilteredWords(trainingclass4))
    filteredwords5=getDict(getFilteredWords(trainingclass5))
    
    filteredwords=[filteredwords1,filteredwords2,filteredwords3,filteredwords4,filteredwords5]
    filteredwordsSum=[sum(filteredwords1.values()),sum(filteredwords2.values()),sum(filteredwords3.values()),sum(filteredwords4.values()),sum(filteredwords5.values())]
    del filteredwords1,filteredwords2,filteredwords3,filteredwords4,filteredwords5

    #for prediction
    pred=[]
    gold=[]
    for i in testingdata:
        gold.append(int(i['overall']))
        pred.append(predict(i['reviewText'],probR,filteredwords,filteredwordsSum))
    
    acc = np.sum(np.equal(gold,pred))/len(gold)
    return acc,pred,gold

########################## (b) ###############################
def partb(trainingdata,classes):
    random=1/len(classes)
    probR=[0,0,0,0,0]
    for i in trainingdata:
        if i['overall']==1.0:
            probR[0]=probR[1]+1
        elif i['overall']==2.0:
            probR[1]=probR[1]+1
        elif i['overall']==3.0:
            probR[2]=probR[2]+1
        elif i['overall']==4.0:
            probR[3]=probR[3]+1
        else:
            probR[4]=probR[4]+1
    majority=max(probR)/sum(probR)
    return random,majority
    
# by randomly guessing any of the class we get the test set accuracy of 1/5 = 0.2 =20%
# by always guessing the class which occurs most frequently we get for class 5
# which is 25932/50109 = 0.5175 = 51.75%

# the algorithm implemented in part a) gives 39.84% more accuracy as compared to random baseline
# and 8.09% more accuracy as compared to majority baseline 


########################## (c) ###############################
def partc(classes,gold,pred):
    cm=confusionMatrix(classes,gold,pred)
    ax= plt.subplot()
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['Class 1','Class 2','Class 3','Class 4', 'Class 5'])
    ax.yaxis.set_ticklabels(['Class 1','Class 2','Class 3','Class 4', 'Class 5'])
    plt.figure(figsize = (10,10))
    sns.heatmap(cm, annot=True,ax=ax)
    diags=cm.diagonal()
    highest_val_cm_class=list(diags).index(max(diags))+1
    return cm,highest_val_cm_class
# this means that class 5 (i.e. row and column 4) has high recall and precision
# most of the classes classified by model are class 5 whcih can be seen from CM

########################## (d) ###############################
def partd(trainingdata,testingdata):
    review=[]
    for i in trainingdata:
        review.append({'rating':i['overall'],'review':i['reviewText']})
    trainingdata=review
    del review
    
    for i in trainingdata:
        i['review']=preprocesswithstopwords(i['review'])
    
    trainingclass1=[]
    trainingclass2=[]
    trainingclass3=[]
    trainingclass4=[]
    trainingclass5=[]
    
    probR=[0,0,0,0,0]
    for i in trainingdata:
        if i['rating']==1.0:
            probR[0]=probR[1]+1
            trainingclass1.append(i)
        elif i['rating']==2.0:
            probR[1]=probR[1]+1
            trainingclass2.append(i)
        elif i['rating']==3.0:
            probR[2]=probR[2]+1
            trainingclass3.append(i)
        elif i['rating']==4.0:
            probR[3]=probR[3]+1
            trainingclass4.append(i)
        else:
            probR[4]=probR[4]+1
            trainingclass5.append(i)


    filteredwords1=getDict(getFilteredWords(trainingclass1))
    filteredwords2=getDict(getFilteredWords(trainingclass2))
    filteredwords3=getDict(getFilteredWords(trainingclass3))
    filteredwords4=getDict(getFilteredWords(trainingclass4))
    filteredwords5=getDict(getFilteredWords(trainingclass5))

    filteredwords=[filteredwords1,filteredwords2,filteredwords3,filteredwords4,filteredwords5]
    filteredwordsSum=[sum(filteredwords1.values()),sum(filteredwords2.values()),sum(filteredwords3.values()),sum(filteredwords4.values()),sum(filteredwords5.values())]
    del filteredwords1,filteredwords2,filteredwords3,filteredwords4,filteredwords5

    pred_nonstop=[]
    gold=[]
    for i in testingdata:
        gold.append(int(i['overall']))
        pred_nonstop.append(predictwithoutstopwords(i['reviewText'],probR,filteredwords,filteredwordsSum))
    
    acc_without_stop = np.sum(np.equal(gold,pred_nonstop))/len(gold)
    cm_stop=confusionMatrix(classes,gold,pred_nonstop)
    return acc_without_stop,pred_nonstop,gold,cm_stop
#best performing model is
# accuracy has increased as compared to model in a) by 4.82%

########################## (e) ###############################
def parte(trainingdata,testingdata):
    review=[]
    for i in trainingdata:
        review.append({'rating':i['overall'],'review':i['reviewText']})
    trainingdata=review
    del review
    
    for i in trainingdata:
        i['review']=preprocesswithstopwords(i['review'])
        
    trainingclass1=[]
    trainingclass2=[]
    trainingclass3=[]
    trainingclass4=[]
    trainingclass5=[]
    
    probR=[0,0,0,0,0]
    for i in trainingdata:
        if i['rating']==1.0:
            probR[0]=probR[1]+1
            trainingclass1.append(i)
        elif i['rating']==2.0:
            probR[1]=probR[1]+1
            trainingclass2.append(i)
        elif i['rating']==3.0:
            probR[2]=probR[2]+1
            trainingclass3.append(i)
        elif i['rating']==4.0:
            probR[3]=probR[3]+1
            trainingclass4.append(i)
        else:
            probR[4]=probR[4]+1
            trainingclass5.append(i)
            
    
    
    filteredbigrams1=getDict(getFilteredBigrams(trainingclass1))
    filteredbigrams2=getDict(getFilteredBigrams(trainingclass2))
    filteredbigrams3=getDict(getFilteredBigrams(trainingclass3))
    filteredbigrams4=getDict(getFilteredBigrams(trainingclass4))
    filteredbigrams5=getDict(getFilteredBigrams(trainingclass5))
    
    filteredwordsBigram=[filteredbigrams1,filteredbigrams2,filteredbigrams3,filteredbigrams4,filteredbigrams5]
    filteredwordsBigramSum=[sum(filteredbigrams1.values()),sum(filteredbigrams2.values()),sum(filteredbigrams3.values()),sum(filteredbigrams4.values()),sum(filteredbigrams5.values())]
    del filteredbigrams1,filteredbigrams2,filteredbigrams3,filteredbigrams4,filteredbigrams5
    
    pred=[]
    gold=[]
    for i in testingdata:
        gold.append(int(i['overall']))
        pred.append(predictwithoutstopwordsBigram(i['reviewText'],filteredwordsBigram,filteredwordsBigramSum,probR))
    
    acc = np.sum(np.equal(gold,pred))/len(gold)
    
    cm=confusionMatrix(classes,gold,pred)
    plt.figure(figsize = (10,10))
    sns.heatmap(cm, annot=True)
    return acc,pred,gold,cm



########################## (f) ###############################
def partf(cm):
    precision=[]
    recall=[]
    f1=[]
    f1Json={}
    for i in classes:
        precision.append(cm[i-1][i-1]/(cm[i-1][i-1]+sum([x for k,x in enumerate(cm[i-1][:]) if k!=i-1])))
        recall.append(cm[i-1][i-1]/(cm[i-1][i-1]+sum([x for k,x in enumerate(cm[:][i-1]) if k!=i-1])))
        if precision[i-1]*recall[i-1] ==0:
            f1.append(0)
            f1Json['Class '+str(i)]=0
        else:
            f=2*precision[i-1]*recall[i-1]/(precision[i-1]+recall[i-1])
            f1.append(f)
            f1Json['Class '+str(i)]=f
    
    macrof1=np.average(f1)
    return f1Json,macrof1
# f1 is more suited as it gives better understanding of both precision and recall

########################## (g) ###############################
def partg(trainingdata,testingdata):
    review=[]
    for i in trainingdata:
        review.append({'rating':i['overall'],'review':i['reviewText'],'summary':i['summary']})
    trainingdata=review
    del review
    
    for i in trainingdata:
        i['review']=preprocess(i['review'])
        i['summary']=preprocess(i['summary'])
        
           
    trainingclass1=[]
    trainingclass2=[]
    trainingclass3=[]
    trainingclass4=[]
    trainingclass5=[]
    
    probR=[0,0,0,0,0]
    for i in trainingdata:
        if i['rating']==1.0:
            probR[0]=probR[1]+1
            trainingclass1.append(i)
        elif i['rating']==2.0:
            probR[1]=probR[1]+1
            trainingclass2.append(i)
        elif i['rating']==3.0:
            probR[2]=probR[2]+1
            trainingclass3.append(i)
        elif i['rating']==4.0:
            probR[3]=probR[3]+1
            trainingclass4.append(i)
        else:
            probR[4]=probR[4]+1
            trainingclass5.append(i)
        
    filteredwords1=getDict(getFilteredWords(trainingclass1))
    filteredwords2=getDict(getFilteredWords(trainingclass2))
    filteredwords3=getDict(getFilteredWords(trainingclass3))
    filteredwords4=getDict(getFilteredWords(trainingclass4))
    filteredwords5=getDict(getFilteredWords(trainingclass5))
    
    sumfilteredwords1=getDict(sumgetFilteredWords(trainingclass1,'summary'))
    sumfilteredwords2=getDict(sumgetFilteredWords(trainingclass2,'summary'))
    sumfilteredwords3=getDict(sumgetFilteredWords(trainingclass3,'summary'))
    sumfilteredwords4=getDict(sumgetFilteredWords(trainingclass4,'summary'))
    sumfilteredwords5=getDict(sumgetFilteredWords(trainingclass5,'summary'))
    
    filteredwords=[filteredwords1,filteredwords2,filteredwords3,filteredwords4,filteredwords5]
    filteredwordsSum=[sum(filteredwords1.values()),sum(filteredwords2.values()),sum(filteredwords3.values()),sum(filteredwords4.values()),sum(filteredwords5.values())]
    del filteredwords1,filteredwords2,filteredwords3,filteredwords4,filteredwords5
    
    sumfilteredwords=[sumfilteredwords1,sumfilteredwords2,sumfilteredwords3,sumfilteredwords4,sumfilteredwords5]
    sumfilteredwordsSum=[sum(sumfilteredwords1.values()),sum(sumfilteredwords2.values()),sum(sumfilteredwords3.values()),sum(sumfilteredwords4.values()),sum(sumfilteredwords5.values())]
    del sumfilteredwords1,sumfilteredwords2,sumfilteredwords3,sumfilteredwords4,sumfilteredwords5
    
    pred=[]
    gold=[]
    for i in testingdata:
        gold.append(int(i['overall']))
        pred.append(sumpredict(i['reviewText'],i['summary'],probR,filteredwords,filteredwordsSum,sumfilteredwords,sumfilteredwordsSum))
    
    acc = np.sum(np.equal(gold,pred))/len(gold)
    return acc,pred,gold


############# main func #####################
if __name__ == '__main__':
    args = read_cli()
    trainingdata,testingdata,classes = getInputData(args.path_of_train_data,args.path_of_test_data)
    if args.part_num=='a':
        acc_a,pred_a,gold_a=parta(trainingdata,testingdata)
        np.save('./acc_a',acc_a)
        np.save('./gold_a',gold_a)
        np.save('./pred_a',pred_a)
        print('accuracy',acc_a)
    elif args.part_num=='b':
        random,majority=partb(trainingdata,classes)
        print('random accuracy',random)
        print('majority accuracy',majority)
        if os.path.exists('./acc_a.npy'):
            acc_a=np.load('./acc_a.npy')
            print('difference of accuracy from part a',abs(majority-acc_a)*100,'%')
        else:
            print("Please run part a first to get comparison of accuracies")
    elif args.part_num=='c':
        if os.path.exists('./gold_a.npy') and os.path.exists('./pred_a.npy'):
            gold_a=np.load('./gold_a.npy')
            pred_a=np.load('./pred_a.npy')
            cm_a,higherCLass=partc(classes,gold_a,pred_a)
            print('confusion matrix',cm_a)
            print('class having higher diagonal value is',higherCLass)
        else:
            print("Please run part a first to get part c running")
    elif args.part_num=='d':
        acc_d,pred_d,gold_d,cm_d=partd(trainingdata,testingdata)
        np.save('./acc_d',acc_d)
        np.save('./gold_d',gold_d)
        np.save('./pred_d',pred_d)
        np.save('./cm_d',cm_d)
        print('accuracy',acc_d)
        print('confusion matrix',cm_d)
    elif args.part_num=='e':
        acc_e,pred_e,gold_e,cm_e=parte(trainingdata,testingdata)
        np.save('./acc_e',acc_e)
        np.save('./gold_e',gold_e)
        np.save('./pred_e',pred_e)
        np.save('./cm_e',cm_e)
        print('accuracy',acc_e)
        print('confusion matrix',cm_e)
    elif args.part_num=='f':
        if os.path.exists('./cm_e.npy'):
            cm_e=np.load('./cm_e.npy')
            f1,macrof1=partf(cm_e)
            print('f1',f1)
            print('macro f1',macrof1)
        else:
            print("Please run part e first to get the appropriate result of part f")
    elif args.part_num=='g':
        acc_g,pred_g,gold_g=partg(trainingdata,testingdata)
        print('accuracy',acc_g)
    else:
        print('Invald part selection.')
    
    