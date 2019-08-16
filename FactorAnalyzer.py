import glob
import numpy as np
import csv
import math
import os
import xml.etree.ElementTree as ET
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.mlab import PCA
from sklearn import mixture
from sklearn import decomposition
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler
from scipy import linalg
from collections import namedtuple
from scipy.stats import multivariate_normal

#Code for Data Modelling
TrainFacePath='D:\\deepayan\\study\\study\\sem 2\\ECE 763\\proj1\\train_face'
TrainNonFacePath='D:\\deepayan\\study\\study\\sem 2\\ECE 763\\proj1\\train_non_face'
TestFacePath='D:\\deepayan\\study\\study\\sem 2\\ECE 763\\proj1\\test_face'
TestNonFacePath='D:\\deepayan\\study\\study\\sem 2\\ECE 763\\proj1\\test_non_face'
train_size=1000
test_size=100
dim=3600
number_of_components=100
threshold=0.5
height=20
width=20
K=10
precison=0.1
correction_factor=10**(-7)
iterations=2

def read_Data(filepath):  
        os.chdir(filepath)
        image=os.listdir()
        X = []
        for img in image:
            img=cv2.imread(img)
            img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            resized_img=cv2.resize(img,(height,width))
            resized_img=resized_img.flatten().tolist()
            X.append(resized_img)
        
        X=np.matrix(X)
        X=np.array(X)
        return X


def perform_PCA(data,number_of_components):     
        pca=PCA(n_components=number_of_components)     
        return (pca.fit_transform(data))

def diagonal_covariance(data):
    s=(number_of_components,number_of_components)
    diag_E=np.zeros(s)
    for k in range(len(data)):
        diag_E[k][k]=data[k][k]
    return (diag_E)

###########################################################
## Fitting the Factor Analyzer
##########################################################

def fit_FactorAnalyzer(data,K,iterations):
    X=np.transpose(data)
    I,D=np.shape(X)
    
    mu=np.mean(X,axis=1)
    phi=np.random.rand(D,K)
    
    X_minus_mu=(np.transpose(X)-np.transpose(mu))
    sig=np.sum(np.square(X_minus_mu),axis=1)
    sig=np.divide(sig,I)
    

    count=0
    while count<=iterations:

        sig_inv=np.absolute(np.diag(np.divide(1,sig)))
        phi=np.array(phi)
        phi_transpose_times_sigma=np.matmul(phi.T,sig_inv)
        temp=np.linalg.inv((np.matmul(phi_transpose_times_sigma,phi))+np.eye(K))
        X_minus_mu=np.array(X_minus_mu)
        E_hi=np.matmul(np.matmul(temp,phi_transpose_times_sigma),X_minus_mu)
        
        E_hi_hitr =[None]*I
        for i in range(I):
            e=E_hi[:,i]
            E_hi_hitr[i]=temp+np.matmul(e,np.transpose(e))
        
        
        s=(D,K)
        phi_1=np.zeros(s)
        x_minus_mu=np.asmatrix(X_minus_mu)
        x_minus_mu=x_minus_mu.T
        for i in range(I):
            a=np.array(x_minus_mu[i,:])
            a=a.T
            b=np.array(E_hi[:,i])
            b=b.T
            c=np.outer(a,b)
            phi_1=phi_1+c
        s=(K,K)    
        phi_2 = np.zeros(s);
        for i in range(I):
            phi_2 = phi_2 + E_hi_hitr[i]
        
        phi_2 = np.asmatrix(np.linalg.inv(phi_2))
        phi_1 =np.matmul(phi_1,phi_2)
        
   #Update sig. 
        s=(1,D)
        sig_diag=np.zeros(s)
        for i in range(I):
            xm=np.transpose(np.array(x_minus_mu[i,:]))
            sig_1=np.multiply(xm,xm)
            a=np.array((np.matmul(phi,E_hi[:,i])))
            a=a.T
            sig_2=np.multiply(a,xm)
            sig_diag = sig_diag + (sig_1 - sig_2)
        
        count=count+1
    
        if(count==iterations):
            break
    return (mu,phi,sig)
        

def evaluate_Model(TrainFacePath,TestFacePath,TrainNonFacePath,TestNonFacePath):
    TF=0
    TN=0
    FP=0
    FN=0
    data=read_Data(TrainFacePath)
    mu,phi,sig=fit_FactorAnalyzer(data,K,iterations)
    
    mu =np.divide(mu,np.max(mu))
    mu_mat =np.reshape(mu,(height,width)) 
    
    
    sig=np.divide(sig,np.max(sig))
    
    
    a=np.matmul(phi,np.transpose(phi))
    c=np.absolute(a+sig)
    c_inv=np.linalg.inv(c)
    
    dataTest=read_Data(TestFacePath)
    predictions1=[0]*test_size
    for tdata in dataTest:
        d=tdata
        d_minus_mu=(d-mu)
        d_minus_mu_squared=np.absolute(np.matmul(d_minus_mu,np.transpose(d_minus_mu)))
        d_minus_mu_squared=d_minus_mu_squared*correction_factor
        pdf=np.absolute(np.exp(-d_minus_mu_squared))
        if(pdf>=threshold):
            predictions1.append(pdf)
        else:
            predictions1.append(pdf)
    TP=predictions1.count(1)
    FN=test_size-TP
    

    print ("True Positive of the model= ", TP)
    print ("False Negative of the model= ", FN)
    
    dataTest=read_Data(TestNonFacePath)
    predictions2=[0]*test_size
    for tdata in dataTest:
        d=tdata
        d_minus_mu=(d-mu)
        d_minus_mu_squared=np.absolute(np.matmul(d_minus_mu,np.transpose(d_minus_mu)))
        d_minus_mu_squared=d_minus_mu_squared*correction_factor
        pdf=np.absolute(np.exp(-d_minus_mu_squared))
        if(pdf>=threshold):
            predictions2.append(pdf)
        else:
            predictions2.append(pdf)
    FP=predictions2.count(1)
    TN=test_size-FP
    print ("False Positive of the model= ", FP)
    print ("True Negative of the model= ", TN)
    a=TP+FP
    PRECISION=(np.true_divide(TP,a))*100
    b=TP+FN
    RECALL=(np.true_divide(TP,b))*100
    c=TP+TN
    d=TP+TN+FP+FN
    ACCURACY=(np.true_divide(c,d))*100
    print ("PRECISON OF MODEL = ",PRECISION)
    print ("RECALL OF MODEL = ",RECALL)
    print ("ACCURACY OF MODEL = ",ACCURACY)
    new_TP1=[0]*test_size
    for i in range(len(predictions1)-test_size,len(predictions1)):
        new_TP1[(len(predictions1)-test_size)-i]=predictions1[i]
    new_FP1=[0]*test_size
    for i in range(len(predictions2)-test_size,len(predictions2)):
        new_FP1[(len(predictions2)-test_size)-i]=predictions2[i]
    TNFN=TN+FN
    FPR=np.absolute(np.divide(FP,TNFN))
    print ("False Positive Rate = ",FPR)
    FPTP=FP+TP
    FNR=np.absolute(np.divide(FN,FPTP))
    print ("False Negative Rate = ",FNR)
    FPFN=FP+FN
    MCR=np.absolute(np.divide(FPFN,test_size))
    print ("Misclassification Rate = ",MCR)
    plot_ROC(np.asarray(new_TP1),np.asarray(new_FP1),test_size)

def plot_ROC(f_nf, f_f,test_size):
    print ("***********PLOTTING ROC*******************")
    predictions = np.append(f_nf, f_f)
    temp1 = [0]*test_size
    temp2 = [1]*test_size
    actual = np.append(temp1,temp2)
    false_positive_rate, true_positive_rate, _ = roc_curve(actual, predictions)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    #print(predictions)
    
    plt.xlabel('False Positive Rate ')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Model_4 Factor Analyser')
    plt.legend(loc="lower right")
    plt.plot(false_positive_rate, true_positive_rate, 'b')
    plt.show()  
            
evaluate_Model(TrainFacePath,TestFacePath,TrainNonFacePath,TestNonFacePath)
