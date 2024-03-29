import os
import xml.etree.ElementTree as ET
import cv2
import glob
import numpy as np
import csv
import math
from matplotlib.mlab import PCA
from sklearn import mixture
from sklearn import decomposition
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import namedtuple
from scipy.stats import multivariate_normal
from sklearn.metrics import roc_curve, auc

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
height=60
width=60
K=3
precison=0.1
correction_factor=10**(-13)
iteration=10

def read_Data(filepath):  
        os.chdir(filepath)
        image=os.listdir()
        X = []
        for img in image:
            img=cv2.imread(img)
            img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            img=cv2.resize(img,(60,60))
            img=img.flatten().tolist()
            X.append(img)
        
        X=np.matrix(X)
        X=np.array(X)
        return X
   

def diagonal_covariance(data):
    s=(number_of_components,number_of_components)
    diag_E=np.zeros(s)
    for k in range(len(data)):
        diag_E[k][k]=data[k][k]
    return (diag_E)

def perform_PCA(data,number_of_components):
        pca=PCA(n_components=number_of_components)
        return (pca.fit_transform(data))

def gaussian(data_vector,mean_vector,covariance_vector):
    
    
    a=np.absolute(np.array(data_vector))
    a_T=a.transpose()
    mean_vector=np.absolute(np.array(mean_vector))
    mean_T=np.absolute(mean_vector.transpose())
    c=np.absolute((a_T-mean_T))
    cov_inv=np.absolute(np.linalg.inv(covariance_vector))
    d=np.absolute(np.matmul(c,cov_inv))
    e=np.absolute(np.matmul(d,np.transpose(c)))
    pdf=np.absolute(np.exp(-e))
    return (pdf)
    
def fit_GMM_Train_Face(data,K,precison):
    X=data
    I,D=np.shape(X)
    
    lambda_vec=[1.0/K]*K
    
    mean_vector=[]
    rand_index=np.random.choice(I,K)
    for k in rand_index:
        row_vec=X[k,:]
        mean_vector.append(row_vec)
    mean_vector=np.array(mean_vector)
    mean_vector=np.divide(mean_vector,I)
     
    dataset_mean=np.mean(X,0)
    s=(D,D)
    dataset_variance=np.zeros(s)
    
    for i in range(I):
        mat=X[i,:]-dataset_mean
        var=(mat*np.transpose(mat))
        dataset_variance=dataset_variance+var
    dataset_variance=np.divide(dataset_variance,I)
    diag_var=diagonal_covariance(dataset_variance)
    
    covariance_matrix_container=[None]*K
    
    for k in range(K):
        covariance_matrix_container[k]=diag_var
      
    
    prev_L=1000000
    counter=0
    while counter<iteration:    
        #Initialization
        s=(I,K)
        l=np.zeros(s)
        r=np.zeros(s)
        
        for k in range(K): 
                
                for i in range(I):
                    mean_vector_k=mean_vector[k,:]
                    data_vector=X[i,:]
                    covariance_k=covariance_matrix_container[k]
                    a=np.absolute(np.transpose(data_vector))
                    b=np.absolute(np.transpose(mean_vector_k))
                    c=(a-b)
                    #b=(data_vector-mean_vector_k)
                    #c=np.matmul(a,b)
                    cov_inv=np.linalg.inv(covariance_k)
                    d=np.matmul(c,cov_inv)
                    e=np.transpose(c)
                    f=np.matmul(d,e)*correction_factor
                    pdf=np.absolute(np.exp(-f))
                    l[i][k]=lambda_vec[k]*pdf
    
                s=np.sum(l,1)
                for i in range(I):
                    r[i][k]=np.divide(l[i][k],s[i])
        sum_along_rows=np.sum(r,0)
        sum_all=np.sum(sum_along_rows)
        
        for k in range(K):
            #Update lambda vector
            lambda_vec[k]=np.divide(sum_along_rows[k],sum_all)
            
            #Update mean vector
            s=(1,D)
            new_mean_vector=np.zeros(s)
            for i in range(I):
                new_mean_vector=new_mean_vector+(r[i,k]*X[i,:])
            mean_vector[k,:]=np.divide(new_mean_vector,sum_along_rows[k])  
            
            #Update the covariance matrix
            s=(D,D)
            new_sigma=np.zeros(s)
            for i in range(I):
                mat=X[i,:] - mean_vector[k,:]
                var=np.matmul(mat,np.transpose(mat))
                var=r[i,k]*var
                new_sigma=new_sigma+var
            new_sigma=np.divide(new_sigma,sum_along_rows[k])
            diag_M_E=diagonal_covariance(new_sigma)
            
            covariance_matrix_container[k]=(diag_M_E)
        
        s=(I,K)
        temp=np.zeros(s)
        for k in range(K):
            for i in range(I):
                mean_vector_k=mean_vector[k,:]
                data_vector=X[i,:]
                covariance_k=covariance_matrix_container[k]
                a=np.absolute(np.transpose(data_vector))
                b=np.absolute(np.transpose(mean_vector_k))
                c=(a-b)
                d=np.matmul(c,covariance_k)
                e=np.transpose(c)
                f=np.matmul(d,e)*correction_factor
                pdf=np.absolute(np.exp(-f))
                temp[i][k]=lambda_vec[k]*pdf
            
        temp=np.sum(temp,1)
        temp=np.log(temp)
        L=np.sum(temp)

        counter=counter+1
        #print ("Value of i",counter)
    return (lambda_vec,mean_vector,covariance_matrix_container)
        
                    
                    
                    
def evaluate_ModelFace(TestFacePath):
    
    dataFace=read_Data(TestFacePath)    
    pca_data1=perform_PCA(dataFace,number_of_components)
    lambda_vec,mean_vec,covar_matrix=fit_GMM_Train_Face(pca_data1,K,precison)
    
    
    TP1=0
    probabilities=[0]*test_size
    for i in range(len(pca_data1)):
        data_vector=pca_data1[i,:]
        out_probs1=[None]*K
        for k in range(K):
            mean_vector=mean_vec[k,:]
            covariance=covar_matrix[k]
            e=gaussian(data_vector,mean_vector,covariance)

            out_probs1[k]=lambda_vec[k]*e
            
        summed_probabilities1=np.sum(out_probs1)
        if(summed_probabilities1>0.5):
            probabilities.append(1)
            TP1=TP1+1

    return TP1,probabilities

    
def evaluate_ModelNonFace(TestNonFacePath):
    
    dataNonFace=read_Data(TestNonFacePath)    
    pca_data1=perform_PCA(dataNonFace,number_of_components)
    lambda_vec,mean_vec,covar_matrix=fit_GMM_Train_Face(pca_data1,K,precison)
    
    FP1=0
    probabilities=[0]*test_size
    for i in range(len(pca_data1)):
        data_vector=pca_data1[i,:]
        out_probs1=[None]*K
        for k in range(K):
            mean_vector=mean_vec[k,:]
            covariance=covar_matrix[k]
            e=gaussian(data_vector,mean_vector,covariance)
            #print ("Output PDF=",e)
            out_probs1[k]=e
            
        summed_probabilities1=np.sum(out_probs1)
        if(summed_probabilities1<0.5):
            probabilities.append(1)
            FP1=FP1+1
    return FP1,probabilities
    
def evaluate_model(TP,FP,probstp,probsfp):
       
       FN=test_size-TP
       TN=test_size-FP
       TP1=evaluate_ModelFace(TestFacePath)
       TP1=np.asarray(TP1[1])
       #print (TP1)
       FP1=evaluate_ModelNonFace(TestNonFacePath)
       FP1=np.asarray(FP1[1])
       new_TP1=[0]*test_size
       for i in range(len(TP1)-test_size,len(TP1)):
           #print (i)
           new_TP1[len(TP1)-1-i]=TP1[i]
       new_FP1=[0]*test_size
       for i in range(len(FP1)-test_size,len(FP1)):
           #print (i)
           new_FP1[len(FP1)-1-i]=FP1[i]
       #PRECISION=np.absolute((TP)/(TP+FP))
       a=TP+FP
       PRECISION=(np.true_divide(TP,a))*100
       b=TP+FN
       RECALL=(np.true_divide(TP,b))*100
       #RECALL=TP/(TP+FN)
       c=TP+TN
       d=TP+TN+FP+FN
       ACCURACY=(np.true_divide(c,d))*100
       print ("PRECISON OF MODEL = ",PRECISION)
       print ("RECALL OF MODEL = ",RECALL)
       print ("ACCURACY OF MODEL = ",ACCURACY)
       TNFN=TN+FN
       FPR=np.absolute(np.divide(FP,TNFN))
       print ("False Positive Rate = ",FPR)
       FPTP=FP+TP
       FNR=np.absolute(np.divide(FN,FPTP))
       print ("False Negative Rate = ",FNR)
       FPFN=FP+FN
       MCR=np.absolute(np.divide(FPFN,test_size))
       print ("Misclassification Rate = ",MCR)
       plot_ROC(np.asarray(new_FP1),np.asarray(new_TP1),test_size)
       
def plot_ROC(f_nf, f_f,test_size):
    
    print ("***********PLOTTING ROC*******************")
    predictions = np.append(f_nf, f_f)
    temp1 = [0]*test_size
    temp2 = [1]*test_size
    actual = np.append(temp1,temp2)
    false_positive_rate, true_positive_rate, _ = roc_curve(actual, predictions)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Model_2 Mixture of Gaussian')
    plt.legend(loc="lower right")
    plt.plot(false_positive_rate, true_positive_rate, 'b')
    plt.show()
    
data=read_Data(TrainFacePath)
pca_data=perform_PCA(data,number_of_components)
TP,probstp=evaluate_ModelFace(TestFacePath)
FP,probsfp=evaluate_ModelNonFace(TestNonFacePath)
evaluate_model(TP,FP,probstp,probsfp)