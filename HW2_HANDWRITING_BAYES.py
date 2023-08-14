#homework 2: handwriting

#step1: read in handwriting dataset and organize
import numpy as np
import numpy.linalg as LA
import scipy.io
mat = scipy.io.loadmat('mnist_all.mat')

import warnings

#turn off warnings
warnings.filterwarnings('ignore')

#define discriminant function
#using generic case, covariance matrices aren't equal
def discriminant(x,COV,mu,Prior):
    #x must be a column vector
    #mu must be a column vector
    
    invCOV = LA.pinv(COV) #pseudo inverse
    [u, s, vh] = LA.svd(COV) 
    
    s = s/s[0]
    detCOV = np.prod(s[s > 0.01])
    
    W = 0.5*invCOV
    w = invCOV@mu
    w0 = 0.5*np.transpose(mu)@invCOV@mu - 0.5*np.log(detCOV) + np.log(Prior)
    
    g = np.transpose(x)@W@x + np.transpose(w)@x + w0
    g = np.reshape(g, ())
    
    return g


#training data and testing data is a 28x28 pixel box with a number handwriting in it
#data is vectorized from 28x28 to a 784 long vector

#only test on first 50 samples from each class
keep = 50

test0 = mat["test0"]
test0 = test0[0:keep,:]

test1 = mat["test1"]
test1 = test1[0:keep,:]

test2 = mat["test2"]
test2 = test2[0:keep,:]

test3 = mat["test3"]
test3 = test3[0:keep,:]

test4 = mat["test4"]
test4 = test4[0:keep,:]

test5 = mat["test5"]
test5 = test5[0:keep,:]

test6 = mat["test6"]
test6 = test6[0:keep,:]

test7 = mat["test7"]
test7 = test7[0:keep,:]

test8 = mat["test8"]
test8 = test8[0:keep,:]

test9 = mat["test9"]
test9 = test9[0:keep,:]

#combines all into one for the loop later to work 
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


#Gather data and calculate mean, COV matrix, prior probability for use in the discriminant functions
train0 = mat["train0"]
train1 = mat["train1"]
train2 = mat["train2"]
train3 = mat["train3"]
train4 = mat["train4"]
train5 = mat["train5"]
train6 = mat["train6"]
train7 = mat["train7"]
train8 = mat["train8"]
train9 = mat["train9"]

#estimate Mean and Covariance matrix of the feature vector for each digit
#using maximum likelihood 

#mean vector: mean of every pixel in the 28x28 image
mean0 = np.reshape(np.mean(train0, axis=0), (1,np.shape(train0)[1]))
mean1 = np.reshape(np.mean(train1, axis=0), (1,np.shape(train1)[1]))
mean2 = np.reshape(np.mean(train2, axis=0), (1,np.shape(train2)[1]))
mean3 = np.reshape(np.mean(train3, axis=0), (1,np.shape(train3)[1]))
mean4 = np.reshape(np.mean(train4, axis=0), (1,np.shape(train4)[1]))
mean5 = np.reshape(np.mean(train5, axis=0), (1,np.shape(train5)[1]))
mean6 = np.reshape(np.mean(train6, axis=0), (1,np.shape(train6)[1]))
mean7 = np.reshape(np.mean(train7, axis=0), (1,np.shape(train7)[1]))
mean8 = np.reshape(np.mean(train8, axis=0), (1,np.shape(train8)[1]))
mean9 = np.reshape(np.mean(train9, axis=0), (1,np.shape(train9)[1]))

#covariance matrix
COV0 = np.cov(train0, rowvar = False)
COV1 = np.cov(train1, rowvar = False)
COV2 = np.cov(train2, rowvar = False)
COV3 = np.cov(train3, rowvar = False)
COV4 = np.cov(train4, rowvar = False)
COV5 = np.cov(train5, rowvar = False)
COV6 = np.cov(train6, rowvar = False)
COV7 = np.cov(train7, rowvar = False)
COV8 = np.cov(train8, rowvar = False)
COV9 = np.cov(train9, rowvar = False)

#calculate prior probabilities
#total number of samples
NumSample = np.shape(train0)[0]+np.shape(train1)[0]+np.shape(train2)[0]+np.shape(train3)[0]+np.shape(train4)[0]+np.shape(train5)[0]+np.shape(train6)[0]+np.shape(train7)[0]+np.shape(train8)[0]+np.shape(train9)[0]
P0 = np.shape(train0)[0] / NumSample
P1 = np.shape(train1)[0] / NumSample
P2 = np.shape(train2)[0] / NumSample
P3 = np.shape(train3)[0] / NumSample
P4 = np.shape(train4)[0] / NumSample
P5 = np.shape(train5)[0] / NumSample
P6 = np.shape(train6)[0] / NumSample
P7 = np.shape(train7)[0] / NumSample
P8 = np.shape(train8)[0] / NumSample
P9 = np.shape(train9)[0] / NumSample

#calculate discriminant fcns and create confusion matrix
#confusion matrix
#rows: actual values, columns: predicted values
Confusion = np.zeros((10,10))
numPixel = np.shape(train0)[1]

begin = 0
end = keep


for j in range(10):
    #samples in training data
    print("Looking at data from test:")
    print(j)
    NumSampleTest = keep
    
    #extract out relevant testing data

    testCurr = testtest[begin:end,:]#needs to be a mx784 matrix
    begin = end
    end = begin + keep
    '''
    try:
        begin = end + 1
        end = begin + TestingLens[j+1]
        Length = end - begin
    except:
        continue
    '''
    
    ConfusionTally = np.zeros((1,10))
    gtest = np.zeros((1,10))
    #print(NumSampleTrain)
    
    for i in range(NumSampleTest): #go through all samples in testj
        #print(i)
        #reshape into correct dimensions
        testing = testCurr[i,:]#np.reshape((testCurr[i,:]),(1,numPixel))
        
        
        gtest[:,0] = discriminant(np.transpose(testCurr[i,:]), COV0, np.transpose(mean0), P0)
        gtest[:,1] = discriminant(np.transpose(testCurr[i,:]), COV1, np.transpose(mean1), P1)
        gtest[:,2] = discriminant(np.transpose(testCurr[i,:]), COV2, np.transpose(mean2), P2)
        gtest[:,3] = discriminant(np.transpose(testCurr[i,:]), COV3, np.transpose(mean3), P3)
        gtest[:,4] = discriminant(np.transpose(testCurr[i,:]), COV4, np.transpose(mean4), P4)
        gtest[:,5] = discriminant(np.transpose(testCurr[i,:]), COV5, np.transpose(mean5), P5)
        gtest[:,6] = discriminant(np.transpose(testCurr[i,:]), COV6, np.transpose(mean6), P6)
        gtest[:,7] = discriminant(np.transpose(testCurr[i,:]), COV7, np.transpose(mean7), P7)
        gtest[:,8] = discriminant(np.transpose(testCurr[i,:]), COV8, np.transpose(mean8), P8)
        gtest[:,9] = discriminant(np.transpose(testCurr[i,:]), COV9, np.transpose(mean9), P9)
        
        maximum = min(gtest[0])
        MaxInd = np.where(gtest[0] == maximum)
        
        #for each class, the confusion "vector" is made. Basically, we add 1 to the indices where a maximum is found
        #in gtest
        #If there's a true positive, it will show up in the diagonal of the "confusion matrix"
        ConfusionTally[:,MaxInd[0][0]] = ConfusionTally[:,MaxInd[0][0]] + 1
       
        
    
    Confusion[j,:] = ConfusionTally #I'm not sure if this is how the confusion matrix is made....

        
        
# Calculate accuracy from confusion matrix
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
    



