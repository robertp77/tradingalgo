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
from sklearn.linear_model import LogisticRegression
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
diabollys=pd.read_csv('5ydia.csv')
bndbollys=pd.read_csv('5ybnd.csv')
dbcbollys=pd.read_csv('5ydbc.csv')
uupbollys=pd.read_csv('5yuup.csv')



bndopen=bndbollys.pop('Open')
bndclose=bndbollys.pop('Close')
diaopen=diabollys.pop('Open')
diaclose=diabollys.pop('Close')
dbcopen=dbcbollys.pop('Open')
dbcclose=dbcbollys.pop('Close')
uupopen=uupbollys.pop('Open')
uupclose=uupbollys.pop('Close')


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


bndupper=bndbollys.pop('UBB(2)')
bndlower=bndbollys.pop('LBB(2)')
diaupper=diabollys.pop('UBB(2)')
dialower=diabollys.pop('LBB(2)')
dbcupper=dbcbollys.pop('UBB(2)')
dbclower=dbcbollys.pop('LBB(2)')
uupupper=uupbollys.pop('UBB(2)')
uuplower=uupbollys.pop('LBB(2)')


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



y=np.zeros((125*10,))
for x in range(0,125):
    y[x]=np.sign(diaclose[10*x]-diaopen[10*x])
    if y[x]==0:
        y[x]=1
for x in range(0,125):
    y[x+125]=np.sign(diaclose[10*x+1]-diaopen[10*x+1])
    if y[x+125]==0:
        y[x+125]=1
for x in range(0,125):
    y[x+125*2]=np.sign(diaclose[10*x+2]-diaopen[10*x+2])
    if y[x+125*2]==0:
        y[x+125*2]=1
for x in range(0,125):
    y[x+125*3]=np.sign(diaclose[10*x+3]-diaopen[10*x+3])
    if y[x+125*3]==0:
        y[x+125*3]=1
for x in range(0,125):
    y[x+125*4]=np.sign(diaclose[10*x+4]-diaopen[10*x+4])
    if y[x+125*4]==0:
        y[x+125*4]=1
for x in range(0,125):
    y[x+125*5]=np.sign(diaclose[10*x+5]-diaopen[10*x+5])
    if y[x+125*5]==0:
        y[x+125*5]=1
for x in range(0,125):
    y[x+125*6]=np.sign(diaclose[10*x+6]-diaopen[10*x+6])
    if y[x+125*6]==0:
        y[x+125*6]=1
for x in range(0,125):
    y[x+125*7]=np.sign(diaclose[10*x+7]-diaopen[10*x+7])
    if y[x+125*7]==0:
        y[x+125*7]=1
for x in range(0,125):
    y[x+125*8]=np.sign(diaclose[10*x+8]-diaopen[10*x+8])
    if y[x+125*8]==0:
        y[x+125*8]=1
for x in range(0,125):
    y[x+125*9]=np.sign(diaclose[10*x+9]-diaopen[10*x+9])
    if y[x+125*9]==0:
        y[x+125*9]=1
X=np.zeros((125*10,16))
n=1
for x in range(0,125):
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
n=2
for x in range(125,125*2):
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
n=3
for x in range(125*2,125*3):
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
n=4
for x in range(125*3,125*4):
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
n=5
for x in range(125*4,125*5):
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
n=6
for x in range(125*5,125*6):
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
n=7
for x in range(125*6,125*7):
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
n=8
for x in range(125*7,125*8):
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
n=9
for x in range(125*8,125*9):
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
n=10
for x in range(125*9,125*10):
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
         #('PCA',  pca),
         ('SVM', SVC(random_state=21, probability=True))]
pipeline = Pipeline(steps)
#'linear', 'rbf', 'poly', 'sigmoid'
# Specify the hyperparameter space
c_space = np.logspace(0, 2, 15) #From Datacamp Scikit Learn Hyperparameter Tuning with GridSearchCV

parameters = {'SVM__kernel':[ 'sigmoid'],
              #'SVM__degree':[3],
              #'SVM__gamma': ['scale', 'auto'],
              #'SVM__gamma': np.logspace(-3, -1, 10),
              'SVM__gamma': [0.024],
              'SVM__C':[8],
              'SVM__decision_function_shape': ['ovo']}
              #'PCA__n_components':np.linspace(0.6, 0.9,10)}
              #'PCA__n_components':[0.5]}
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
sell=0
for x in range(250,0,-1):
    # Create train and test sets
    #X_train, X_test, y_train, y_testt = train_test_split(X,y,test_size=0.0005, random_state=x)
    
    # Instantiate the GridSearchCV object: cv
    cv = GridSearchCV(pipeline, parameters, cv=2,n_jobs=20)
    #cv = RandomizedSearchCV(pipeline, parameters, cv=5, n_jobs=20, random_state=20)
    X_test=np.reshape(np.array(X[x,:]),(1,-1))
    X_train=np.delete(X,slice(x+1),0)
    y_train=np.delete(y,slice(x+1),0)
    y_test=np.reshape(np.array(y[x]),(1,))
    # Fit to the training set
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
    else:
        if justsold==False:
            sell=diaopen[x]
            justsold=True
            justbought=False
        elif justsold==True and sell!=0:
            justsold=False
    if justsold==True and sell!=0:
        money+=(sell-buy)
    '''
    if cv.score(X_test, y_test)!=0:
        if y_pred==-1:
            money+=(diaopen[x]-diaclose[x])   
    else:
        if y_pred==-1:
            money+=(diaopen[x]-diaclose[x])  
    '''
    results.append(cv.score(X_test, y_test))
    print(money)
print(money+diaopen[0]-buy)
nres=np.array(results)
print(np.mean(nres))

