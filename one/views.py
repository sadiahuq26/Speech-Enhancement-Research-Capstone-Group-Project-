from django.shortcuts import render
from django.conf import settings
from django.conf import urls
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from joblib import load
import os
# Create your views here.
global mono_xgc,bi_xgc,mono_label_encoder,bili_label_encoder
mono_xgc=''
bi_xgc=''
mono_label_encoder=''
bili_label_encoder=''

global r
r=True

def home(request):
    global r
    global mono_xgc,bi_xgc,mono_label_encoder,bili_label_encoder
    print(r)
    if r:
        #mono()
        mono2()
        #bili()
        bili2()
        r=False
    else:
        print('NOISE NOISE GO AWAY')
    print(type(bili_label_encoder))
    return render(request,'index.html')

def categories(request):
    audio_path=os.path.join(settings.STATICFILES_DIRS[0],'audio')
    a=os.listdir(audio_path)
    s=''
    for i in a:
       s+=f'<div class="box"><span>{i}</span><audio controls><source src="/static/audio/{i}" type="audio/wav"></audio></div>'
    return render(request,'categories.html',{'s':s})

def blog(request):
    return render(request,'blog.html')

def contact(request):
    return render(request,'contact.html')

def bilingual(request):
    person=7
    if request.method=='POST':
        rating=list()
        for i in range(1,person+1):
            rating.append(int(request.POST.get('Person '+str(i))))
        inp= np.reshape(rating, (1, -1))
        randomTest = bi_xgc.predict(inp)
        res=bili_label_encoder.inverse_transform(randomTest)
        print(res)
        return render(request,'bilingual-result.html',{'res':res[0]})
        
    return render(request,'bilingual.html',{'person':range(1,person+1),'rating':range(1,11)})


def monolingual(request):
    person=11
    if request.method=='POST':
        rating=list()
        for i in range(1,person+1):
            rating.append(int(request.POST.get('Person '+str(i))))
        inp= np.reshape(rating, (1, -1))
        randomTest = mono_xgc.predict(inp)
        res=mono_label_encoder.inverse_transform(randomTest)
        print(res)
        return render(request,"monolingual-result.html",{'res':res[0]})
    return render(request,'monolingual.html',{'person':range(1,person+1),'rating':range(1,11)})




def mono():
    global mono_xgc,bi_xgc,mono_label_encoder,bili_label_encoder
    dataset = pd.read_csv('one/cse400datasetMonoLingual.csv')
    dataset=dataset.iloc[1:121]
    dataset=dataset.dropna(axis=1, how='all')
    mono_label_encoder = preprocessing.LabelEncoder()
    dataset['stim code']= mono_label_encoder.fit_transform(dataset['stim code'])
    train, test = train_test_split(dataset, test_size=0.3) 
    train_features = train.iloc[:,:11] #X_train
    train_target = train["stim code"] #Y_train
    test_features = test.iloc[:,:11] #X_test 
    test_target = test["stim code"]  #Y_test
    std_scaler = StandardScaler()
    train_features = std_scaler.fit_transform(train_features)
    test_features = std_scaler.fit_transform(test_features)
    mono_xgc=XGBClassifier(n_estimators=100,max_depth= 9,subsample=1.0,colsample_bytree=0.9,random_state=42)
    mono_xgc.fit(train_features,train_target)
    y_pred3 = mono_xgc.predict(test_features)   
    print('MONO:',y_pred3)
    
def bili():
    global mono_xgc,bi_xgc,mono_label_encoder,bili_label_encoder
    
    dataset = pd.read_csv('one/cse400datasetBilingual.csv')
    dataset=dataset.iloc[1:121]
    dataset=dataset.dropna(axis=1, how='all')
    bili_label_encoder = preprocessing.LabelEncoder()
    dataset['stim code']= bili_label_encoder.fit_transform(dataset['stim code'])
    train, test = train_test_split(dataset, test_size=0.3) 
    train_features = train.iloc[:,:7] #X_train
    train_target = train["stim code"] #Y_train
    test_features = test.iloc[:,:7] #X_test 
    test_target = test["stim code"]  #Y_test
    std_scaler = StandardScaler()
    train_features = std_scaler.fit_transform(train_features)
    test_features = std_scaler.fit_transform(test_features)
    bi_xgc=XGBClassifier(n_estimators=100,max_depth= 9,subsample=1.0,colsample_bytree=0.9,random_state=42)
    bi_xgc.fit(train_features,train_target)
    y_pred3 = bi_xgc.predict(test_features)
    print('BILI:',y_pred3)
    
def mono2():
    global mono_xgc,bi_xgc,mono_label_encoder,bili_label_encoder
    
    mono_xgc=load('one/mono.joblib')
    mono_label_encoder=load('one/mono_label.joblib')
    
def bili2():
    global mono_xgc,bi_xgc,mono_label_encoder,bili_label_encoder
    
    bi_xgc=load('one/bili.joblib')
    bili_label_encoder=load('one/bili_label.joblib')