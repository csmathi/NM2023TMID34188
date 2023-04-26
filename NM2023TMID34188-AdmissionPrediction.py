#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[34]:


#read_csv is a pandas function to read csv files
data=pd.read_csv('E:\\NMDS\Admission_Predict.csv')
data.head()


# In[37]:


data.isnull().sum()


# In[5]:


data.isnull().any()


# In[35]:


data.info()


# In[36]:


data.shape


# In[6]:


#let us rename the column chance of Admit because it has trainling space
data=data.rename(columns={'chance of Admit':'chance of Admit'})


# In[7]:


data.describe()


# In[8]:


sns.distplot(data['GRE Score'])


# In[38]:


plt.figure(figsize=(10,7))
sns.heatmap(data.corr(),annot=True)


# In[9]:


sns.pairplot(data=data,hue='Research',markers=["^","v"],palette='inferno')
                                                   


# In[24]:


sns.scatterplot(x='University Rating',y='CGPA',data=data,color='red',s=100)


# In[25]:


category = ['GRE Score','TOEFL Score','University Rating','SOP','LOR','CGPA','Research','chance of Admit']
color = ['yellowgreen','gold','lightskyblue','pink','red','purple','orange','gray']
start = True
for i in np.arange(4): 
  fig = plt.figure(figsize=(14,8))
  plt.subplot2grid((4,2),(i,0)) 
  data[category[2*i]].hist(color=color[2*i],bins=10)
  plt.title(category[2*i])
  plt.subplot2grid((4,2),(i,1))
  data[category[2*i+1]].hist(color=color[2*i+1],bins=10)
  plt.title(category[2*i+1])
plt.subplots_adjust(hspace = 0.7,wspace = 0.2)
plt.show()


# In[26]:


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
x=sc.fit_transform(x)
x


# In[27]:


x=data.iloc[:,0:7].values
x


# In[28]:


y=data.iloc[:,7:].values
y


# In[11]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=101)
#random_state acts as the seed for the random number generator during the split


# In[29]:


y_train=(y_train>0.5)
y_train


# In[30]:


y_test=(y_test>0.5)


# In[31]:


from sklearn.linear_model.logistic import LogisticRegression
cls =LogisticRegression(random_state =0)

lr=cls.fit(x_train,y_train)

c:\Users\Tulasi\anaconda3\lib\site.packages\sklearn\utils\validation.py:760: DataConversionwarn  
array was expected.please change the shape of y to(n_samples,), for example using ravel().
 y = column_or_1d(y,warn=true)
    
y_pred =lr.predict(x_test)
y_pred


# In[12]:


#Libraries to train Neural network
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense,Activation,Dropout
from tensorfrom.keras.optimizers import Adam


# In[32]:


#Initialize the model
model=Keras.Sequential()

#Add input layer
model.add(Dense(7,activation ='relu',input_dim=7))

#Add hidden layer
model.add(Dense(7,activation ='relu'))

#Add output layer
model.add(Dense(1,activation ='linear'))

model.summary()

model: "sequential"
model.summary()
model: "sequential"


# In[14]:


model.fit(x_train, y_train, batch_size = 20, epochs = 100)


# In[15]:


model.compile(loss = 'binary_crossentropy', optimizer = 'adam',metics = ['accuracy'])


# In[16]:


model.fit(x_train, y_train, batch_size = 20, epochs = 100)


# In[17]:


from sklearn.metrics import accuracy_score


 
#make predictions on the training data
train_predictions = model.predict(x_train)

print(train_predictions)


# In[18]:


# Get the training accuracy
train_acc = model.evaluate(x_train, y_train,verbos=0)[1]

print(train_acc)


# In[19]:


#Get the test accuracy
test_acc = model.evaluate(x_test, y_test, verbose=0)[1]
print(test_acc)


# In[20]:


print(classification report(v test,pred))


# In[21]:


pred=model.predict(x_test)
pred = (pred>0.5)
pred


# In[39]:



from sklearn.metrics import accuracy_score,recall_score,roc_auc_score,confusion_matrix
print("\nAccuracy_score: %f"  %(accuracy_score(y_test,y_pred)*100))
print("Recall Score: %f"  %(recall_score(y_test,y_pred)*100))
print("ROC_Score: %f"  %(roc_auc_score(y_test,y_pred)*100))
print(confusion_matrix(y_test,y_pred))


# In[ ]:


#save the model in h5 format
model.save('model5.h')


# In[ ]:


import numpy as np
from flask import Flask,request,jsonify, render_teplate
import pickle
app=Flask(__name__)
#import necessary libraries
from.tensorflow.keras.models import load_model
#model=pickle.load(open('university.pkl','rb'))


# In[ ]:


#load model trained model
load.model('model.h5')


# In[ ]:


@pp.route('/')
def.home():
    retun render_template('Demo2.html')

