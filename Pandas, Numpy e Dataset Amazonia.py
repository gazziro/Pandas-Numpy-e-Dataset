#!/usr/bin/env python
# coding: utf-8

# In[126]:


import matplotlib.pyplot as plt
import numpy as np


# In[135]:


x = [-1, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 1]
y = [-1.13, -0.57, -0.2, 0.5, 0.49, 1.49, 1.64, 2.17, 2.64, 2.95]


# In[137]:


plt.figure(figsize=(10,5))
plt.plot(x,y,'o-', label="dados originais")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.show()


# In[129]:


x,y = np.array(x).reshape(-1, 1), np.array(y).reshape(-1,1)
X = np.hstack((x, np.ones(x.shape)))

beta = np.linalg.pinv(X).dot(y)
print("a estimado:",beta[0][0])
print("b estimado:",beta[1][0])


# In[130]:


plt.figure(figsize=(10,5))
plt.plot(x,y,'o', label="dados originais")
plt.plot(x,X.dot(beta), label="regressão linear")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.show()


# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv("https://pycourse.s3.amazonaws.com/temperature.csv")


# In[3]:


df


# In[4]:


excel_file = pd.ExcelFile("https://pycourse.s3.amazonaws.com/temperature.xlsx")


# In[5]:


df2 = pd.read_excel(excel_file, sheet_name="Sheet1")
df2


# In[6]:


df['temperatura']


# In[7]:


df.iloc[3:7]


# In[8]:


df.loc[3, 'temperatura']


# In[9]:


df[df['classification'] == 'quente']


# In[10]:


df.loc[df['classification'] == 'quente', 'temperatura']


# In[11]:


df['date']


# In[14]:


type(df['date'][0])
#Não é date 


# In[15]:


df['temperatura']


# In[16]:


type(df['temperatura'][0])
#Não é int


# In[17]:


df[['date', 'classification']]


# In[18]:


df.iloc[:]


# In[24]:


get_ipython().run_line_magic('pinfo', 'df.iloc')


# In[19]:


df.iloc[:1]


# In[20]:


df.iloc[:,1]


# In[21]:


df.iloc[:,0]


# In[22]:


df.loc[:,'temperatura']


# In[23]:


df.loc[:,'date']


# In[24]:


df.loc[:,['temperatura', 'date']]


# In[25]:


df['date'] = pd.to_datetime(df['date'])


# In[26]:


type(df['date'])


# In[27]:


df.dtypes


# In[28]:


df


# In[30]:


df = df.set_index('date')


# In[31]:


df


# In[32]:


cond = df['temperatura'] >= 25
cond


# In[33]:


df[cond]


# In[34]:


df[df.index <= '2020-02-01']


# In[35]:


df.plot()


# In[41]:


df['classification'].value_counts().plot.bar(figsize=(10,5), rot=0, grid=True)


# In[36]:


df['classification'].value_counts().plot.pie(figsize=(10,5),autopct='%1.1f%%', grid=True)


# In[43]:


df.groupby(by='classification')


# In[44]:


df.groupby(by='classification').mean()


# In[45]:


df.groupby(by='classification').sum()


# In[46]:


df


# In[47]:


df.drop('temperatura', axis=1)


# In[80]:


df


# In[81]:


df2 = df.copy()


# In[82]:


df2


# In[51]:


df2.drop('temperatura', axis=1)


# In[52]:


df2.head()


# In[53]:


df2 = df2.drop('temperatura', axis=1)


# In[54]:


df2


# In[55]:


df3 = df.drop('temperatura', axis=1, inplace=True)


# In[56]:


df3


# In[57]:


print(df3)


# In[58]:


df3 = df.copy()


# In[59]:


df3.drop('temperatura', axis=1, inplace=True)


# In[61]:


df


# In[62]:


df3


# In[39]:


from sklearn.preprocessing import LabelEncoder


# In[40]:


x, y = df[['temperatura']].values, df[['classification']].values


# In[41]:


df


# In[42]:


df


# In[43]:


df['date'] = pd.to_datetime(df['date'])


# In[95]:


df


# In[44]:


df.dtypes


# In[45]:


x, y = df[['temperatura']].values, df[['classification']].values
print('x:\n ', x)
print('y:\n ',y)


# In[87]:


get_ipython().run_line_magic('pinfo', 'LabelEncoder')


# In[46]:


le = LabelEncoder()
y = le.fit_transform(y.ravel())
print("y:\n ", y)


# In[100]:


from sklearn.linear_model import LogisticRegression


# In[101]:


clf = LogisticRegression()
clf.fit(x,y)


# In[102]:


x_test = np.linspace(start=0, stop=45, num=100).reshape(-1,1)
y_pred = clf.predict(x_test)


# In[103]:


print(y_pred)


# In[104]:


y_pred = le.inverse_transform(y_pred)
print(y_pred)


# In[105]:


print(x_test)


# In[110]:


output = {'new_temp': x_test.ravel(),'new_class':y_pred.ravel()}
output = pd.DataFrame(output)


# In[112]:


output.tail()


# In[113]:


output.head()


# In[115]:


output.info()


# In[116]:


output.describe()


# In[119]:


output['new_class'].value_counts().plot.bar(figsize=(10,5), title="novos valores gerados", rot=0, grid=True)


# In[123]:


output.boxplot(by='new_class', figsize=(20,10))


# In[124]:


def classify_temp():
    ask = True
    while ask:
        temp = input("Insira a temperatura (graus): ")
        temp = np.array(float(temp)).reshape(-1,1)
        class_temp = clf.predict(temp)
        class_temp = le.inverse_transform(class_temp)
        print(f"a classificacao da temperatura {temp.ravel()[0]} é:", class_temp[0])
        ask = input("nova classificacao (y/n): ")=='y'


# In[125]:


classify_temp()


# In[ ]:




