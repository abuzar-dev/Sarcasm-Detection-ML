#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


dataset=pd.read_csv('Sarcasm Detection.tsv',delimiter='\t',quoting=3)


# In[3]:


import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
corpus=[]
for i in range(0,1000):
    review =re.sub('[^a-zA-Z]',' ',dataset['Review'][i]) #we are replacing all the punctions with white spaces
    review=review.lower()  #converting the review into lower case
    review=review.split() # converting our statement into list of words
    #ps=PorterStemmer()
    wordnet=WordNetLemmatizer()
    #here 'word' is a variable which will contain all the words from review list one by one
    #review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    all_stopwords=stopwords.words('english') #will collect all the stop words
    all_stopwords.remove('not') #remove not form the stopword
#   review=[ps.stem(word) for word in review if not word in set(all_stopwords)]
    review=[wordnet.lemmatize(word) for word in review if not word in set(all_stopwords)]
    #if and only if the word is not present in the stopword will it be allowed to pass on the object of the stemmer class
    review=' '.join(review) # joining all the words of the review list back together to create the cleaned statement review
    corpus.append(review)


# In[4]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
#here while creating the object of the countvectorizer class we need to input one important parameter
#and that parameter is max_features which decided after we get the total number of columns
x=cv.fit_transform(corpus).toarray()
y=dataset.iloc[:,-1].values


# In[5]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=325)


# In[6]:


from sklearn.decomposition import PCA
pca = PCA(n_components=750)# since we do not know how many eigenvectors
# are need we keep the value of n components = None so that we can the
# eigenvalues of all the evectors to figure out the best ones
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
explained_variance = pca.explained_variance_ratio_
print(explained_variance)
# after all the evalues are obtained select the number of evectors and replace
# the value of n_components by that number


# In[7]:


from sklearn.svm import SVC
classifier = SVC(kernel='linear',random_state=24)
classifier.fit(x_train,y_train)


# In[8]:


y_pred = classifier.predict(x_test)
print(y_pred)


# In[9]:


#Creating the confusion matrix and calculating the accuracy score
from sklearn.metrics import plot_confusion_matrix, accuracy_score, confusion_matrix
acc = accuracy_score(y_test,y_pred)
print(acc)
cm = confusion_matrix(y_test,y_pred)
print(cm)
plot_confusion_matrix(classifier,x_test,y_test,cmap=plt.cm.Blues)
plt.show()


# In[ ]:




