#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt  
import pandas as pd 


# In[2]:


data_set=pd.read_csv("C:/Users/DELL/Downloads/Salary_Data.csv")


# In[3]:


data_set.head()


# In[4]:


x= data_set.iloc[:, :-1].values  
y= data_set.iloc[:, 1].values 


# In[6]:


# Splitting the dataset into training and test set.  
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 1/3, random_state=0) 


# In[7]:


y_test


# In[8]:


#Fitting the Simple Linear Regression model to the training dataset  
from sklearn.linear_model import LinearRegression  
regressor= LinearRegression()  
regressor.fit(x_train, y_train)


# In[9]:


#Prediction of Test and Training set result  
y_pred= regressor.predict(x_test)  
y_pred


# In[10]:


x_pred= regressor.predict(x_train)  
x_pred


# In[11]:


plt.scatter(x_train, y_train, color="green")   
plt.plot(x_train, x_pred, color="red")    
plt.title("Salary vs Experience (Training Dataset)")  
plt.xlabel("Years of Experience")  
plt.ylabel("Salary(In Rupees)")  
plt.show()  


# In[12]:


#visualizing the Test set results  
plt.scatter(x_test, y_test, color="blue")   
plt.plot(x_train, x_pred, color="red")    
plt.title("Salary vs Experience (Test Dataset)")  
plt.xlabel("Years of Experience")  
plt.ylabel("Salary(In Rupees)")  
plt.show()


# In[13]:


# Root Mean Squared Error on training dataset
from sklearn.metrics import mean_squared_error
rmse_train = mean_squared_error(y_test,y_pred)**(0.5)
print('\nRMSE on train dataset : ', rmse_train)


# In[ ]:




