# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 15:47:35 2020

@author: arobu
"""

import pickle
import numpy as np
import pandas as pd
import requests
import os
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline 
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV 
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
import time
from sklearn.metrics import plot_confusion_matrix, roc_curve, roc_auc_score
import tensorflow
from tensorflow import keras
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
'''
f = open('list_pickle3.pkl', 'rb') #<-- USE THIS FOR 92%

[X,y] = pickle.load(f)
diabollys=pd.read_csv('bollingers.csv')
bndbollys=pd.read_csv('bndbollys.csv')
dbcbollys=pd.read_csv('dbcbollys.csv')
uupbollys=pd.read_csv('uupbollys.csv')

diabollys.drop(diabollys.index[0],axis=0,inplace=True)

bndbollys.drop(bndbollys.index[1],axis=0,inplace=True)
bndbollys.drop(bndbollys.index[0],axis=0,inplace=True)
bndbollys.index = range(len(bndbollys.index))
dbcbollys.drop(dbcbollys.index[1],axis=0,inplace=True)
dbcbollys.drop(dbcbollys.index[0],axis=0,inplace=True)
uupbollys.drop(uupbollys.index[1],axis=0,inplace=True)
uupbollys.drop(uupbollys.index[0],axis=0,inplace=True)
diabollys.index = range(len(diabollys.index))
dbcbollys.index = range(len(dbcbollys.index))
uupbollys.index = range(len(uupbollys.index))
for x in range(496,516):
    dbcbollys.drop(dbcbollys.index[len(dbcbollys.index)-1],axis=0,inplace=True)
    uupbollys.drop(uupbollys.index[len(uupbollys.index)-1],axis=0,inplace=True)
for x in range(0,496):
    if y[x]==0:
        y[x]=1
bndupper=bndbollys.pop('UBB(2)')
bndlower=bndbollys.pop('LBB(2)')
diaupper=diabollys.pop('UBB(2)')
dialower=diabollys.pop('LBB(2)')
dbcupper=dbcbollys.pop('UBB(2)')
dbclower=dbcbollys.pop('LBB(2)')
uupupper=uupbollys.pop('UBB(2)')
uuplower=uupbollys.pop('LBB(2)')


for x in range(0,496):
    for xx in range(0,13):
        X[x,xx]=(X[x,xx]-dialower[x])/(diaupper[x]-dialower[x])
    for xx in range(13,16):
        X[x,xx]=(X[x,xx]-bndlower[x])/(bndupper[x]-bndlower[x])
    for xx in range(16,19):
        X[x,xx]=(X[x,xx]-dbclower[x])/(dbcupper[x]-dbclower[x])
    for xx in range(19,22):
        X[x,xx]=(X[x,xx]-uuplower[x])/(uupupper[x]-uuplower[x])
'''
diabollys=pd.read_csv('5yspynew.csv')
bndbollys=pd.read_csv('5ybndnew.csv')
dbcbollys=pd.read_csv('5ydbcnew.csv')
uupbollys=pd.read_csv('5yuupnew.csv')



bndopen=bndbollys.pop('Open')
bndclose=bndbollys.pop('Close')
diaopen=diabollys.pop('Open')
diaclose=diabollys.pop('Close')
dbcopen=dbcbollys.pop('Open')
dbcclose=dbcbollys.pop('Close')
uupopen=uupbollys.pop('Open')
uupclose=uupbollys.pop('Close')

'''
diaopen.loc[-1]=299.38
diaopen.index=diaopen.index+1
diaopen = diaopen.sort_index() 
diaclose.loc[-1]=300.6
diaclose.index=diaclose.index+1
diaclose = diaclose.sort_index() 
diaopen.loc[-1]=300.74
diaopen.index=diaopen.index+1
diaopen = diaopen.sort_index() 
diaclose.loc[-1]=302.59
diaclose.index=diaclose.index+1
diaclose = diaclose.sort_index() 
bndopen.loc[-1]=88.13
bndopen.index=bndopen.index+1
bndopen = bndopen.sort_index() 
bndclose.loc[-1]=88.1
bndclose.index=bndclose.index+1
bndclose = bndclose.sort_index() 
bndopen.loc[-1]=88
bndopen.index=bndopen.index+1
bndopen = bndopen.sort_index() 
bndclose.loc[-1]=87.87
bndclose.index=bndclose.index+1
bndclose = bndclose.sort_index() 
dbcopen.loc[-1]=13.93
dbcopen.index=dbcopen.index+1
dbcopen = dbcopen.sort_index() 
dbcclose.loc[-1]=13.98
dbcclose.index=dbcclose.index+1
dbcclose = dbcclose.sort_index() 
dbcopen.loc[-1]=14.01
dbcopen.index=dbcopen.index+1
dbcopen = dbcopen.sort_index() 
dbcclose.loc[-1]=14.04
dbcclose.index=dbcclose.index+1
dbcclose = dbcclose.sort_index() 
uupopen.loc[-1]=24.44
uupopen.index=uupopen.index+1
uupopen = uupopen.sort_index() 
uupclose.loc[-1]=24.44
uupclose.index=uupclose.index+1
uupclose = uupclose.sort_index() 
uupopen.loc[-1]=24.41
uupopen.index=uupopen.index+1
uupopen = uupopen.sort_index() 
uupclose.loc[-1]=24.47
uupclose.index=uupclose.index+1
uupclose = uupclose.sort_index() 
'''

bndupper=bndbollys.pop('UBB(2)')
bndlower=bndbollys.pop('LBB(2)')
diaupper=diabollys.pop('UBB(2)')
dialower=diabollys.pop('LBB(2)')
dbcupper=dbcbollys.pop('UBB(2)')
dbclower=dbcbollys.pop('LBB(2)')
uupupper=uupbollys.pop('UBB(2)')
uuplower=uupbollys.pop('LBB(2)')

'''
diaupper.loc[-1]=304.65
diaupper.index=diaupper.index+1
diaupper = diaupper.sort_index() 
dialower.loc[-1]=285.67
dialower.index=dialower.index+1
dialower = dialower.sort_index() 
diaupper.loc[-1]=304.6
diaupper.index=diaupper.index+1
diaupper = diaupper.sort_index() 
dialower.loc[-1]=287.6
dialower.index=dialower.index+1
dialower = dialower.sort_index() 
bndupper.loc[-1]=88.61
bndupper.index=bndupper.index+1
bndupper = bndupper.sort_index() 
bndlower.loc[-1]=87.46
bndlower.index=bndlower.index+1
bndlower = bndlower.sort_index() 
bndupper.loc[-1]=88.59
bndupper.index=bndupper.index+1
bndupper = bndupper.sort_index() 
bndlower.loc[-1]=87.45
bndlower.index=bndlower.index+1
bndlower = bndlower.sort_index() 
dbcupper.loc[-1]=14.19
dbcupper.index=dbcupper.index+1
dbcupper = dbcupper.sort_index() 
dbclower.loc[-1]=13.05
dbclower.index=dbclower.index+1
dbclower = dbclower.sort_index() 
dbcupper.loc[-1]=14.22
dbcupper.index=dbcupper.index+1
dbcupper = dbcupper.sort_index() 
dbclower.loc[-1]=13.1
dbclower.index=dbclower.index+1
dbclower = dbclower.sort_index() 
uupupper.loc[-1]=25.225
uupupper.index=uupupper.index+1
uupupper = uupupper.sort_index() 
uuplower.loc[-1]=24.54
uuplower.index=uuplower.index+1
uuplower = uuplower.sort_index() 
uupupper.loc[-1]=25.24
uupupper.index=uupupper.index+1
uupupper = uupupper.sort_index() 
uuplower.loc[-1]=24.47
uuplower.index=uuplower.index+1
uuplower = uuplower.sort_index() 
'''

dt=124
y=np.zeros((dt*10,))
for xx in range(0,10):
    for x in range(0,dt):
        if x==0 and xx==0:
            y[x+xx*dt]=np.sign(diaclose[10*x+xx]-diaopen[10*x+xx])
        else:
            y[x+xx*dt]=np.sign(diaopen[10*x+xx-1]-diaopen[10*x+xx])
        if y[x+xx*dt]==0:
            y[x+xx*dt]=1
        elif y[x+xx*dt]==-1:
            y[x+xx*dt]=0

X=np.zeros((dt*10,16))
n=1
for xx in range(0,10):
    n=xx+1
    for x in range(dt*xx,dt*(xx+1)):
        X[x,0]=(diaopen[n-1]-dialower[n])/(diaupper[n]-dialower[n])
        X[x,1]=(((diaopen[n]+diaclose[n])/2)-dialower[n])/(diaupper[n]-dialower[n])
        X[x,2]=(((diaopen[n+1]+diaclose[n+1])/2)-dialower[n])/(diaupper[n]-dialower[n])
        X[x,3]=(((diaopen[n+2]+diaclose[n+2])/2)-dialower[n])/(diaupper[n]-dialower[n])
        X[x,4]=(((diaopen[n+3]+diaclose[n+3])/2)-dialower[n])/(diaupper[n]-dialower[n])
        
        X[x,5]=(((diaopen[n+4]+diaclose[n+4])/2)-dialower[n])/(diaupper[n]-dialower[n])
        X[x,6]=(((diaopen[n+5]+diaclose[n+5])/2)-dialower[n])/(diaupper[n]-dialower[n])
        X[x,7]=(((diaopen[n+6]+diaclose[n+6])/2)-dialower[n])/(diaupper[n]-dialower[n])
        X[x,8]=(((diaopen[n+7]+diaclose[n+7])/2)-dialower[n])/(diaupper[n]-dialower[n])
        X[x,9]=(((diaopen[n+8]+diaclose[n+8])/2)-dialower[n])/(diaupper[n]-dialower[n])
        '''
        xs=[]
        xs.append((((diaopen[n+4]+diaclose[n+4])/2)-dialower[n])/(diaupper[n]-dialower[n]))
        xs.append((((diaopen[n+4]+diaclose[n+4])/2)-dialower[n])/(diaupper[n]-dialower[n]))
        xs.append((((diaopen[n+4]+diaclose[n+4])/2)-dialower[n])/(diaupper[n]-dialower[n]))
        xs.append((((diaopen[n+4]+diaclose[n+4])/2)-dialower[n])/(diaupper[n]-dialower[n]))
        xs.append((((diaopen[n+4]+diaclose[n+4])/2)-dialower[n])/(diaupper[n]-dialower[n]))
        X[x,5]=np.mean(np.array(xs))
        '''
        bnds=[]
        bnds.append((bndopen[n]-bndlower[n])/(bndupper[n]-bndlower[n]))
        bnds.append((bndclose[n]-bndlower[n])/(bndupper[n]-bndlower[n]))
        bnds.append((bndopen[n+1]-bndlower[n])/(bndupper[n]-bndlower[n]))
        bnds.append((bndclose[n+1]-bndlower[n])/(bndupper[n]-bndlower[n]))
        bnds.append((bndopen[n+2]-bndlower[n])/(bndupper[n]-bndlower[n]))
        bnds.append((bndclose[n+2]-bndlower[n])/(bndupper[n]-bndlower[n]))
        bnds.append((bndopen[n+3]-bndlower[n])/(bndupper[n]-bndlower[n]))
        bnds.append((bndclose[n+3]-bndlower[n])/(bndupper[n]-bndlower[n]))
        bnds.append((bndopen[n+4]-bndlower[n])/(bndupper[n]-bndlower[n]))
        bnds.append((bndclose[n+4]-bndlower[n])/(bndupper[n]-bndlower[n]))
        X[x,10]=np.mean(np.array(bnds))
        bnds.append((bndopen[n+5]-bndlower[n])/(bndupper[n]-bndlower[n]))
        bnds.append((bndclose[n+5]-bndlower[n])/(bndupper[n]-bndlower[n]))
        bnds.append((bndopen[n+6]-bndlower[n])/(bndupper[n]-bndlower[n]))
        bnds.append((bndclose[n+6]-bndlower[n])/(bndupper[n]-bndlower[n]))
        bnds.append((bndopen[n+7]-bndlower[n])/(bndupper[n]-bndlower[n]))
        bnds.append((bndclose[n+7]-bndlower[n])/(bndupper[n]-bndlower[n]))
        bnds.append((bndopen[n+8]-bndlower[n])/(bndupper[n]-bndlower[n]))
        bnds.append((bndclose[n+8]-bndlower[n])/(bndupper[n]-bndlower[n]))
        X[x,11]=np.mean(np.array(bnds))
        dbcs=[]
        dbcs.append((dbcopen[n]-dbclower[n])/(dbcupper[n]-dbclower[n]))
        dbcs.append((dbcclose[n]-dbclower[n])/(dbcupper[n]-dbclower[n]))
        dbcs.append((dbcopen[n+1]-dbclower[n])/(dbcupper[n]-dbclower[n]))
        dbcs.append((dbcclose[n+1]-dbclower[n])/(dbcupper[n]-dbclower[n]))
        dbcs.append((dbcopen[n+2]-dbclower[n])/(dbcupper[n]-dbclower[n]))
        dbcs.append((dbcclose[n+2]-dbclower[n])/(dbcupper[n]-dbclower[n]))
        dbcs.append((dbcopen[n+3]-dbclower[n])/(dbcupper[n]-dbclower[n]))
        dbcs.append((dbcclose[n+3]-dbclower[n])/(dbcupper[n]-dbclower[n]))
        dbcs.append((dbcopen[n+4]-dbclower[n])/(dbcupper[n]-dbclower[n]))
        dbcs.append((dbcclose[n+4]-dbclower[n])/(dbcupper[n]-dbclower[n]))
        X[x,12]=np.mean(np.array(dbcs))
        dbcs.append((dbcopen[n+5]-dbclower[n])/(dbcupper[n]-dbclower[n]))
        dbcs.append((dbcclose[n+5]-dbclower[n])/(dbcupper[n]-dbclower[n]))
        dbcs.append((dbcopen[n+6]-dbclower[n])/(dbcupper[n]-dbclower[n]))
        dbcs.append((dbcclose[n+6]-dbclower[n])/(dbcupper[n]-dbclower[n]))
        dbcs.append((dbcopen[n+7]-dbclower[n])/(dbcupper[n]-dbclower[n]))
        dbcs.append((dbcclose[n+7]-dbclower[n])/(dbcupper[n]-dbclower[n]))
        dbcs.append((dbcopen[n+8]-dbclower[n])/(dbcupper[n]-dbclower[n]))
        dbcs.append((dbcclose[n+8]-dbclower[n])/(dbcupper[n]-dbclower[n]))
        X[x,13]=np.mean(np.array(dbcs))
        uups=[]
        uups.append((uupopen[n]-uuplower[n])/(uupupper[n]-uuplower[n]))
        uups.append((uupclose[n]-uuplower[n])/(uupupper[n]-uuplower[n]))
        uups.append((uupopen[n+1]-uuplower[n])/(uupupper[n]-uuplower[n]))
        uups.append((uupclose[n+1]-uuplower[n])/(uupupper[n]-uuplower[n]))
        uups.append((uupopen[n+2]-uuplower[n])/(uupupper[n]-uuplower[n]))
        uups.append((uupclose[n+2]-uuplower[n])/(uupupper[n]-uuplower[n]))
        uups.append((uupopen[n+3]-uuplower[n])/(uupupper[n]-uuplower[n]))
        uups.append((uupclose[n+3]-uuplower[n])/(uupupper[n]-uuplower[n]))
        uups.append((uupopen[n+4]-uuplower[n])/(uupupper[n]-uuplower[n]))
        uups.append((uupclose[n+4]-uuplower[n])/(uupupper[n]-uuplower[n]))
        X[x,14]=np.mean(np.array(uups))
        uups.append((uupopen[n+5]-uuplower[n])/(uupupper[n]-uuplower[n]))
        uups.append((uupclose[n+5]-uuplower[n])/(uupupper[n]-uuplower[n]))
        uups.append((uupopen[n+6]-uuplower[n])/(uupupper[n]-uuplower[n]))
        uups.append((uupclose[n+6]-uuplower[n])/(uupupper[n]-uuplower[n]))
        uups.append((uupopen[n+7]-uuplower[n])/(uupupper[n]-uuplower[n]))
        uups.append((uupclose[n+7]-uuplower[n])/(uupupper[n]-uuplower[n]))
        uups.append((uupopen[n+8]-uuplower[n])/(uupupper[n]-uuplower[n]))
        uups.append((uupclose[n+8]-uuplower[n])/(uupupper[n]-uuplower[n]))
        X[x,15]=np.mean(np.array(uups))
        n+=10

pca=PCA()

steps = [('scaler', StandardScaler()),
         ('PCA',  pca),
         ('SVM', SVC(random_state=21, probability=True))]
pipeline = Pipeline(steps)
#'linear', 'rbf', 'poly', 'sigmoid'
# Specify the hyperparameter space
c_space = np.logspace(-2, 2, 15) #From Datacamp Scikit Learn Hyperparameter Tuning with GridSearchCV

parameters = {'SVM__kernel':[ 'sigmoid'],
                #'SVM__kernel':[ 'poly'],
              #'SVM__degree':[3],
              #'SVM__gamma': ['scale', 'auto'],
              'SVM__gamma': [0.02],
              #'SVM__gamma': [0.024],
              'SVM__C':[8],
              'SVM__decision_function_shape': ['ovo'],
              #'PCA__n_components':np.linspace(0.6, 0.9,10)}
              'PCA__n_components':[0.25]}
'''
steps = [('scaler', StandardScaler()),
         ('PCA',  pca),
         ('LR', LogisticRegression(random_state=21))]
pipeline = Pipeline(steps)
#'linear', 'rbf', 'poly', 'sigmoid'
# Specify the hyperparameter space
c_space = np.logspace(0, 2, 15) #From Datacamp Scikit Learn Hyperparameter Tuning with GridSearchCV

parameters = {'LR__solver': ['saga'],
                'LR__penalty':['l2'],
                'LR__C':[1],
              'PCA__n_components':[0.95]}
'''
results=[]
money=0
justbought=False
justsold=True
alreadysold=False
sell=0
monies=[]
buys=[]
sells=[]
# Fit to the training set
'''
model = keras.Sequential([
keras.layers.Flatten(input_shape=(16,)),  # input layer (1)
keras.layers.Dense(8, activation='relu'),
keras.layers.Dense(2, activation='sigmoid') # output layer (3)
])
model.compile(optimizer='adam',
          loss='sparse_categorical_crossentropy',
          metrics=['accuracy'])
model.fit(X_train, y_train, epochs=800) 
'''
for x in range(250,0,-1):
# Create train and test sets
#X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.0005, random_state=x)

# Instantiate the GridSearchCV object: cv
    cv = GridSearchCV(pipeline, parameters, cv=2,n_jobs=20)
    #cv = RandomizedSearchCV(pipeline, parameters, cv=5, n_jobs=20, random_state=20)
    #X_test=np.reshape(np.array(X[0:250,:]),(250,16))
    X_test=np.reshape(np.array(X[x,:]),(1,-1))
    y_test=np.array(y[x])
    # Fit to the training set
    #cv.fit(X_train, y_train)
    X_train=np.delete(X, slice(x+1), axis = 0)
    y_train=np.delete(y,slice(x+1),axis=0)
    cv.fit(X_train, y_train)
    # Predict the labels of the test set: y_pred
    y_pred = cv.predict(X_test)
    #print(cv.predict(np.reshape(np.array([1.625,1.625,-0.70,1.2,0.06,-0.07,-0.5,-0.6,-0.04,-0.14,0.28,0.05,-0.09,-0.16,-0.1,-0.05,0,-0.02,-0.05,-0.1,-0.07,-0.02]),(1,-1))))
    # Compute and print metrics

    if y_pred==1:
        if justbought==False:
            buy=diaopen[x]
            justbought=True
            justsold=False
            alreadysold=False
            buys.append(buy)
        else:
            buys.append(0)
        sells.append(0)
    else:
        if justsold==False and alreadysold==False:
            sell=diaopen[x]
            justsold=True
            alreadysold=True
            justbought=False
            sells.append(sell)
        elif justsold==True and sell!=0:
            alreadysold=True
            justsold=False
            sells.append(0)
        else:
            sells.append(0)
        buys.append(0)
    if justsold==True and sell!=0:
        money+=(sell-buy)
    '''
    if np.argmax(y_pred)==0:
        money+=(diaopen[x]-diaclose[x])  
    if np.argmax(y_pred)==1 and y_test==1:
        results.append(1)
    elif np.argmax(y_pred)==0 and y_test==1:
        results.append(0)
    elif np.argmax(y_pred)==0 and y_test==0:
        results.append(1)
    else:
        results.append(0)
    '''
    if y_pred==1 and y_test==1:
        results.append(1)
    elif y_pred==0 and y_test==1:
        results.append(0)
    elif y_pred==0 and y_test==0:
        results.append(1)
    else:
        results.append(0)
    '''
    if y_pred==0:
        money+=(diaopen[x]-diaclose[x]) 
    '''
    print(x)
    print(money)
    monies.append(money)
print(money+diaopen[0]-buy)
nres=np.array(results)
print(np.mean(nres))
plt.plot(np.arange(0,250,1),np.array(diaopen[250:0:-1]))
plt.plot(np.arange(0,250,1),monies)
plt.plot(np.arange(0,250,1),buys,'g.')
plt.plot(np.arange(0,250,1),sells,'r.')