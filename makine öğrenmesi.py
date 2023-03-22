import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
from sklearn.metrics import f1_score, classification_report
from spacy.tokenizer import Tokenizer
from spacy.lang.tr import Turkish
#from simpletransformers.classification import classificationmodel
import re
import nltk


#Öğrenme Setinin Tanımlanması

ogrenme = pd.read_csv("Ogrenme_Seti.csv")
print(ogrenme)


x = ogrenme.loc[:, ["TWEET"]]
y = ogrenme.loc[:, ["TOPIC"]]



#Verilerin Ölçeklenmesi
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


#Topiclerin sayısal değere dönüştürülmesi
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
labels=le.fit_transform(ogrenme.TOPIC)



from sklearn.feature_extraction.text import CountVectorizer
max_features=500
count_vectorizer=CountVectorizer(max_features=max_features)
X=count_vectorizer.fit_transform(ogrenme.TWEET).toarray()


#Verilerin Eğitim ve Test için Bölünmesi
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)


X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

np.ravel(y_train)



#Tahmin yapmak için tüm verileri içeren dosyanın tanımlanması

veriler = pd.read_csv("veri_seti.csv")

print(veriler)


#Verilerin Vektöre Dönüştürülmesi
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
labelsVeriler=le.fit_transform(veriler.TWEETLER)


from sklearn.feature_extraction.text import CountVectorizer
max_features=500
count_vectorizer=CountVectorizer(max_features=max_features)
XVeriler=count_vectorizer.fit_transform(veriler.TWEETLER).toarray()


#LogRegression Modeli

from sklearn.linear_model import LogisticRegression

  
logr = LogisticRegression(random_state = 0)
logr.fit(X_train, y_train)
y_pred = logr.predict(XVeriler)
print(y_pred)


df = pd.DataFrame(y_pred, columns = ['topic'])
print(df)

df.to_excel('tahmin_logr.xlsx')


#SupportVectorMachine Modeli
from sklearn.svm import SVC
svc = SVC(kernel = "linear")
svc.fit(X_train, y_train)

y_pred = svc.predict(XVeriler)
print(y_pred)

df = pd.DataFrame(y_pred, columns = ['topics'])
print(df)

df.to_excel("tahmin_svc.xlsx")



#NaiveBayes Modeli
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

gnb.fit(X_train, y_train)

y_pred = gnb.predict(XVeriler)
print(y_pred)

df = pd.DataFrame(y_pred, columns = ['topics'])

df.to_excel("tahmin_gnb.xlsx")








