# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 15:16:08 2023

@author: emmay
"""
import numpy as np
import numpy.linalg as LA
import scipy.io
mat = scipy.io.loadmat('mnist_all.mat')

import warnings

#turn off warnings
warnings.filterwarnings('ignore')


def Sigmoid(z):
    
    sig = 1/(1+ np.exp(-z))
    
    return sig

'''
def ObjFcn(theta,x,y):
    [numSample, numFeature] = np.shape(x)
    
    cost = 0
    for i in range(numSample):
        xvect = np.reshape(x[i,:], (numFeature,1)) # (785x1) 
        predict = np.transpose(theta)@xvect
        cost = cost + (-y[i,:]*np.log(Sigmoid(predict)) - (1-y[i,:])*np.log(1- Sigmoid(predict)))
    
    cost = cost/numSample
    return cost
'''

def Gradient(theta, x, y):
    #y: ground truth, column vector
    # column of whatver the number actually should be 
    
    #theta*x: prediction
    
    #x: input feature vector (testing data) COLUMN VECTOR (reshaped in loop)
    #theta: COLUMN VECTOR
    
    [numSample, numFeature] = np.shape(x)
    
    grad = np.zeros((numFeature,1)) 
    #grad = []
    for i in range(numSample):
        xvect = np.reshape(x[i,:], (numFeature,1)) # (785x1) 
        predict = (1/(1-np.exp(-1*np.transpose(theta)@xvect)))
        grad = grad + (predict - y[i,:])*xvect #(785x1)
    
    
    
    return grad

def gradDescent(x,y):
    print(gradDescent)
    #theta = np.ones((np.shape(train0)[1],1))
    theta = np.random.normal(0,1, size = (np.shape(train0)[1],1))
    #eps = 0.01 #stopping criteria 
    #stopCrit = 100 #set large so it can enter the loop
    learnRate = 0.001
    #cost = []
    for i in range(1000):
        #calculate gradient
        gradient = Gradient(theta, x, y)
        #calculte cost function
        #cost.append(ObjFcn(theta, x, y))
        theta_p = theta - learnRate*gradient #(785x1)
        #print(learnRate*gradient)
    
        stopCrit = LA.norm(theta - theta_p)     
        print(stopCrit)
        theta = theta_p
        
    return theta



#organize data
keep = 50

test0 = mat["test0"]
test0 = test0[0:keep,:]
test0 = test0/255  
test0 = np.hstack((np.ones((len(test0),1)), test0)) #(5923 x 785)

test1 = mat["test1"]
test1 = test1[0:keep,:]
test1 = test1/255  
test1 = np.hstack((np.ones((len(test1),1)), test1)) #(5923 x 785


test2 = mat["test2"]
test2 = test2[0:keep,:]
test2 = test2/255  
test2 = np.hstack((np.ones((len(test2),1)), test2)) #(5923 x 785

test3 = mat["test3"]
test3 = test3[0:keep,:]
test3 = test3/255  
test3 = np.hstack((np.ones((len(test3),1)), test3)) #(5923 x 785

test4 = mat["test4"]
test4 = test4[0:keep,:]
test4 = test4/255  
test4 = np.hstack((np.ones((len(test4),1)), test4)) #(5923 x 785

test5 = mat["test5"]
test5 = test5[0:keep,:]
test5 = test5/255  
test5 = np.hstack((np.ones((len(test5),1)), test5)) #(5923 x 785

test6 = mat["test6"]
test6 = test6[0:keep,:]
test6 = test6/255  
test6 = np.hstack((np.ones((len(test6),1)), test6)) #(5923 x 785

test7 = mat["test7"]
test7 = test7[0:keep,:]
test7 = test7/255  
test7 = np.hstack((np.ones((len(test7),1)), test7)) #(5923 x 785

test8 = mat["test8"]
test8 = test8[0:keep,:]
test8 = test8/255  
test8 = np.hstack((np.ones((len(test8),1)), test8)) #(5923 x 785

test9 = mat["test9"]
test9 = test9[0:keep,:]
test9 = test9/255  
test9 = np.hstack((np.ones((len(test9),1)), test9)) #(5923 x 785

testtest = np.vstack((test0,test1,test2,test3,test4,test5,test6,test7,test8,test9))
t0Len = np.shape(test0)[0]
t1Len = np.shape(test1)[0]
t2Len = np.shape(test2)[0]
t3Len = np.shape(test3)[0]
t4Len = np.shape(test4)[0]
t5Len = np.shape(test5)[0]
t6Len = np.shape(test6)[0]
t7Len = np.shape(test7)[0]
t8Len = np.shape(test8)[0]
t9Len = np.shape(test9)[0]

TestingLens = [t0Len, t1Len, t2Len, t3Len, t4Len, t5Len, t6Len, t7Len, t8Len, t9Len]

#add zeros to training data so logistic regression works

train0 = mat["train0"]
train0 = train0/255  
train0 = np.hstack((np.ones((len(train0),1)), train0)) #(5923 x 785)

#randomly pick samples from other classes, give label "0"
train1 = mat["train1"]
train1 = train1/255  
train1 = np.hstack((np.ones((len(train1),1)), train1)) #(5923 x 785)

train2 = mat["train2"]
train2 = train2/255  
train2 = np.hstack((np.ones((len(train2),1)), train2)) #(5923 x 785)

train3 = mat["train3"]
train3 = train3/255  
train3 = np.hstack((np.ones((len(train3),1)), train3)) #(5923 x 785)

train4 = mat["train4"]
train4 = train4/255  
train4 = np.hstack((np.ones((len(train4),1)), train4)) #(5923 x 785)

train5 = mat["train5"]
train5 = train5/255  
train5 = np.hstack((np.ones((len(train5),1)), train5)) #(5923 x 785)

train6 = mat["train6"]
train6 = train6/255  
train6 = np.hstack((np.ones((len(train6),1)), train6)) #(5923 x 785)

train7 = mat["train7"]
train7 = train7/255  
train7 = np.hstack((np.ones((len(train7),1)), train7)) #(5923 x 785)

train8 = mat["train8"]
train8 = train8/255  
train8 = np.hstack((np.ones((len(train8),1)), train8)) #(5923 x 785)

train9 = mat["train9"]
train9 = train9/255  
train9 = np.hstack((np.ones((len(train9),1)), train9)) #(5923 x 785)


#Prepare data for 1 vs all
train0Vall = np.vstack((train0, train1[0:100], train2[0:100], train3[0:100], train4[0:100], train5[0:100], train6[0:100], train7[0:100], train8[0:100], train9[0:100]))
y0 = np.vstack((np.ones((np.shape(train0)[0],1)), np.zeros((900,1))))

train1Vall = np.vstack((train1, train0[0:100], train2[0:100], train3[0:100], train4[0:100], train5[0:100], train6[0:100], train7[0:100], train8[0:100], train9[0:100]))
y1 = np.vstack((np.ones((np.shape(train1)[0],1)), np.zeros((900,1))))

train2Vall = np.vstack((train2, train0[0:100], train1[0:100], train3[0:100], train4[0:100], train5[0:100], train6[0:100], train7[0:100], train8[0:100], train9[0:100]))
y2 = np.vstack((np.ones((np.shape(train2)[0],1)), np.zeros((900,1))))

train3Vall = np.vstack((train3, train0[0:100], train1[0:100], train2[0:100], train4[0:100], train5[0:100], train6[0:100], train7[0:100], train8[0:100], train9[0:100]))
y3 = np.vstack((np.ones((np.shape(train3)[0],1)), np.zeros((900,1))))

train4Vall = np.vstack((train4, train0[0:100], train1[0:100], train2[0:100], train3[0:100], train5[0:100], train6[0:100], train7[0:100], train8[0:100], train9[0:100]))
y4 = np.vstack((np.ones((np.shape(train4)[0],1)), np.zeros((900,1))))

train5Vall = np.vstack((train5, train0[0:100], train1[0:100], train2[0:100], train3[0:100], train4[0:100], train6[0:100], train7[0:100], train8[0:100], train9[0:100]))
y5 = np.vstack((np.ones((np.shape(train5)[0],1)), np.zeros((900,1))))

train6Vall = np.vstack((train6, train0[0:100], train1[0:100], train2[0:100], train3[0:100], train4[0:100], train5[0:100], train7[0:100], train8[0:100], train9[0:100]))
y6 = np.vstack((np.ones((np.shape(train6)[0],1)), np.zeros((900,1))))

train7Vall = np.vstack((train7, train0[0:100], train1[0:100], train2[0:100], train3[0:100], train4[0:100], train5[0:100], train6[0:100], train8[0:100], train9[0:100]))
y7 = np.vstack((np.ones((np.shape(train7)[0],1)), np.zeros((900,1))))

train8Vall = np.vstack((train8, train0[0:100], train1[0:100], train2[0:100], train3[0:100], train4[0:100], train5[0:100], train6[0:100], train7[0:100], train9[0:100]))
y8 = np.vstack((np.ones((np.shape(train8)[0],1)), np.zeros((900,1))))

train9Vall = np.vstack((train9, train0[0:100], train1[0:100], train2[0:100], train3[0:100], train4[0:100], train5[0:100], train6[0:100], train7[0:100], train8[0:100]))
y9 = np.vstack((np.ones((np.shape(train9)[0],1)), np.zeros((900,1))))
#normalize data. convert from zero to 255 --> zero to 1


# train Model with testing data

#go through each class and find weights associated with them

theta0 = gradDescent(train0Vall,y0)    
theta1 = gradDescent(train1Vall,y1)
theta2 = gradDescent(train2Vall,y2)
theta3 = gradDescent(train3Vall,y3)
theta4 = gradDescent(train4Vall,y4)
theta5 = gradDescent(train5Vall,y5)
theta6 = gradDescent(train6Vall,y6)
theta7 = gradDescent(train7Vall,y7)
theta8 = gradDescent(train8Vall,y8)
theta9 = gradDescent(train9Vall,y9)
 
#go through each set of testing data, and classify

#sigmoid function
Confusion = np.zeros((10,10))

begin = 0
end = keep

for j in range(10): #just for testing
    
    testCurr = testtest[begin:end,:]#needs to be a mx784 matrix
    begin = end
    end = begin + keep
   
    [numSample, numFeature] = np.shape(testCurr)
    ConfusionTally = np.zeros((1,10))

    for i in range(numSample):
        print(i)
        logreg = np.zeros((1,10))
        xvect = np.reshape(testCurr[i,:], (numFeature,1)) #what is the class of the current sample?
        logreg[:,0] = Sigmoid(np.transpose(theta0)@xvect) #these are all just zero no matter what I do 
        logreg[:,1] = Sigmoid(np.transpose(theta1)@xvect)
        logreg[:,2] = Sigmoid(np.transpose(theta2)@xvect)
        logreg[:,3] = Sigmoid(np.transpose(theta3)@xvect)
        logreg[:,4] = Sigmoid(np.transpose(theta4)@xvect)
        logreg[:,5] = Sigmoid(np.transpose(theta5)@xvect)
        logreg[:,6] = Sigmoid(np.transpose(theta6)@xvect)
        logreg[:,7] = Sigmoid(np.transpose(theta7)@xvect)
        logreg[:,8] = Sigmoid(np.transpose(theta8)@xvect)
        logreg[:,9] = Sigmoid(np.transpose(theta9)@xvect)
        
        maximum = max(logreg[0])
        MaxInd = np.where(logreg[0] == maximum)
        if len(MaxInd[0]) > 1:
            MaxInd = np.random.choice(MaxInd[0])
        else:
            MaxInd = MaxInd[0]
            
        ConfusionTally[:,MaxInd] = ConfusionTally[:,MaxInd] + 1
       #MaxInd = np.where(gtest[0] == maximum)
        #choose maximum
        #if conflict, randomly choose
    
    Confusion[j,:] = ConfusionTally #I'm not sure if this is how the confusion matrix is made....

        
#calculate for every class
#TP: True positive
#TN: True negative
#FP: False positive
#FN: False negative
dim = len(Confusion)
TPstore = np.zeros((dim,1))
FNstore = np.zeros((dim,1))
FPstore = np.zeros((dim,1))
TNstore = np.zeros((dim,1))
AccurStore = np.zeros((dim,1))

for i in range(dim):
    TPstore[i] = Confusion[i,i]
    
    beforei_R = Confusion[i,0:i]
    afteri_R = Confusion[i,i+1:dim]
    beforei_C = Confusion[0:i,i]
    afteri_C = Confusion[i+1:dim,i]
    temp = np.delete(Confusion, obj = i, axis=0)
    AllXcept = np.delete(temp, obj = i , axis=1)
    
    FNstore[i] = np.sum(beforei_R) + np.sum(afteri_R)
    FPstore[i] = np.sum(beforei_C) + np.sum(afteri_C)
    TNstore[i] = np.sum(AllXcept)
    
    AccurStore[i] = (TPstore[i] + TNstore[i])/(TPstore[i] + TNstore[i] + FPstore[i] + FNstore[i])
    


