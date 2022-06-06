#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from langdetect import DetectorFactory
from langdetect import detect
from langdetect import detect_langs
import googletrans
from googletrans import Translator


# In[2]:


#import dataset
cosmeticScoreDS = pd.read_csv('trainDS.csv')


# In[3]:


print(googletrans.LANGUAGES)


# In[4]:


#getting only usefull data from import dataset
cosmeticScoreDS = cosmeticScoreDS.iloc[:,[4,16,14,11]].values


# In[5]:


#identify missing data
data = pd.DataFrame(cosmeticScoreDS)
cosmeticScoreDS = data.dropna(axis = 0,how='any')
X = cosmeticScoreDS.iloc[:,:-1].values


# In[6]:


#stored data which of rating field
rating = X[:,-1]
Y = cosmeticScoreDS.iloc[:,-1].values


# In[7]:


#encoding dataset
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y = le.fit_transform(Y)


# In[8]:


#data visualization
cosmeticScoreDS.head(-1)


# In[9]:


#visualize type of X and Y
print(X)
print(Y)


# In[10]:


#clean dataset
import re
import nltk
from nltk.corpus import stopwords

#reduce words in their root form
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
all_stopwords = stopwords.words('english')

# Removing some stopwords which have significance effect in building this model
rem = [ 'just', 'too', 'very', 'no', 'nor', 'only', 'own', 'same', 'again', 'against', 'but', 'not', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', 
       "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', 
       "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", 'don', "don't",]
for s in rem:
  all_stopwords.remove(s)

def find_clean_text(temp):
  temp = re.sub('[^a-zA-Z]', ' ', temp)
  temp = temp.lower()
  temp = temp.split()
  temp = [ps.stem(word) for word in temp if not word in set(all_stopwords)]
  temp = ' '.join(temp)
  return temp


# In[11]:


#concatanating both title and detailed review
corpus = []
for i in range(X.shape[0]):
  temp = X[i][0] + ' ' + X[i][1]
  temp = find_clean_text(temp)
  corpus.append(temp)


# In[12]:


#create bag of word model
cv = CountVectorizer(max_features = 3000)
X = cv.fit_transform(corpus).toarray()


# In[13]:


# Adding rating in the matrix of feature X
rating = rating.reshape(rating.shape[0],1)
X = np.append(X,rating,axis=1)


# In[14]:


#split dataset into test set and train set
pos_x = []
pos_y = []
neg_x = []
neg_y = []
for i in range(X.shape[0]):
  if Y[i]==1:
    pos_x.append(X[i])
    pos_y.append(Y[i])
  else:
    neg_x.append(X[i])
    neg_y.append(Y[i])

X_train1, X_test1, Y_train1, Y_test1 = train_test_split(pos_x, pos_y, test_size = 0.20)
X_train, X_test, Y_train, Y_test = train_test_split(neg_x, neg_y, test_size = 0.20)

for i in range(len(X_train1)):
  X_train.append(X_train1[i])
  Y_train.append(Y_train1[i])
for i in range(len(X_test1)):
  X_test.append(X_test1[i])
  Y_test.append(Y_test1[i])
X_train = np.array(X_train)
X_test = np.array(X_test)
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)


# In[15]:


#Training the multinomial naive bayes(MNB) model on the Training set
classifier = MultinomialNB()
classifier.fit(X_train, Y_train)


# In[16]:


#Making the Confusion Matrix  ---using SciKit-learn metrics
print('Result on training set :')
print('Confusion matrix :')
print(confusion_matrix(Y_train,classifier.predict(X_train)))
print('accuracy : ',accuracy_score(Y_train, classifier.predict(X_train)))

print('Result on test set :')
Y_pred = classifier.predict(X_test)
print('Confusion matrix :')
print(confusion_matrix(Y_test, Y_pred))
print('accuracy',accuracy_score(Y_test, Y_pred))


# In[17]:


#Making Prediction on some reviews from test set
random.seed(13)
translator = Translator()

index = random.randrange(cosmeticScoreDS.shape[0])
result = translator.translate(cosmeticScoreDS[1][index])
print('Title :',cosmeticScoreDS[0][index],'\nReview :',result.text,'\nRating :',cosmeticScoreDS[2][index])
print("True value :", cosmeticScoreDS[3][index])
print("Prediction :",classifier.predict(np.append(cv.transform([find_clean_text(cosmeticScoreDS[0][index]+' '+cosmeticScoreDS[1][index])]).toarray(),[[cosmeticScoreDS[3][index]]],axis=1)))


# In[18]:


#calculating the accuracy
accuracy_score(Y_test,Y_pred)


# In[19]:


import pickle
from sklearn.svm import SVC 
svm_model_pkl = open('PScoreRating_model.pkl', 'wb')
pickle.dump(cosmeticScoreDS, svm_model_pkl)
svm_model_pkl.close()


# In[20]:


cosmeticScoreDS.to_csv('pscorerating')


# In[ ]:




