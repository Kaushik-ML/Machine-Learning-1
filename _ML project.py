#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # loading the data

# In[8]:


med_insurance_df = pd.read_csv("insurance.csv")


# In[9]:


med_insurance_df.head(2)


# In[10]:


med_insurance_df.info()


# # Exploration of data

# In[11]:


med_insurance_df['region'].unique()


# In[12]:


med_insurance_df.groupby('region').max()['charges']


# In[13]:


med_insurance_df.groupby('region').mean()['charges']


# In[14]:


pd.get_dummies(med_insurance_df['region'])


# In[15]:


med_insurance_df['northeast_region'] = pd.get_dummies(med_insurance_df['region'])['northeast']
med_insurance_df['southeast_region'] = pd.get_dummies(med_insurance_df['region'])['southeast']
med_insurance_df['male'] = pd.get_dummies(med_insurance_df['sex'])['male']
med_insurance_df['smoker'] = pd.get_dummies(med_insurance_df['smoker'])['yes']
med_insurance_df.head(2)


# In[16]:


print("Maximun ",med_insurance_df.groupby('male').max()['bmi'])
print("Mean ",med_insurance_df.groupby('male').mean()['bmi'])


# In[17]:


print("Maximun ",med_insurance_df.groupby('smoker').max()['bmi'])
print("Mean ",med_insurance_df.groupby('smoker').mean()['bmi'])


# In[18]:


print("Maximun ",med_insurance_df.groupby('male').max()['charges'])
print("Mean ",med_insurance_df.groupby('male').mean()['charges'])


# In[19]:


print("Maximun ",med_insurance_df.groupby('smoker').max()['charges'])
print("Mean ",med_insurance_df.groupby('smoker').mean()['charges'])


# # Visualizing data

# In[21]:


plt.figure(figsize=(10,6))
sns.heatmap(med_insurance_df.corr(), annot=True, cmap='viridis')


# In[22]:


sns.scatterplot(x='age', y='charges', data=med_insurance_df, hue='smoker')


# In[23]:


sns.scatterplot(x='age', y='charges', data=med_insurance_df, hue='male')


# In[24]:


sns.countplot(x='smoker', data=med_insurance_df, hue='male')


# In[25]:


sns.countplot(x='smoker', data=med_insurance_df, hue='region')


# In[26]:


sns.countplot(x='male', data=med_insurance_df)


# # Drop unnecessary data

# In[27]:


med_insurance_df.columns


# In[28]:


for col in ['sex', 'children', 'region', 'male']:
  if col in med_insurance_df.columns:
    med_insurance_df.drop(col, axis=1, inplace=True)


# # Splitting the data for training and testing.

# In[29]:


X = med_insurance_df.drop('charges', axis=1)
y = med_insurance_df['charges']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# # Linear regression

# In[31]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)

pred = lm.predict(X_test)


# # Model evaluation

# In[32]:


print(lm.intercept_)


# In[33]:


df = pd.DataFrame(lm.coef_, index=X.columns, columns=['Coefficient'])
df


# In[34]:


plt.scatter(y_test, pred)
plt.xlabel("y_test")
plt.ylabel("pred")
plt.title("True value vs. Predicted")


# In[35]:


from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, pred))
print('MSE:', metrics.mean_squared_error(y_test, pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred)))


# In[ ]:




