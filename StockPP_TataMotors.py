#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


# In[2]:


start = '2012-01-01'
end = '2022-12-21'
stock = 'TATAMOTORS.NS'

data = yf.download(stock, start, end)


# In[3]:


data.reset_index(inplace=True)


# In[4]:


data


# In[5]:


ma_100_days = data.Close.rolling(100).mean()


# In[6]:


plt.figure(figsize=(8,6))
plt.plot(ma_100_days, 'r')
plt.plot(data.Close, 'g')
plt.show()


# In[7]:


ma_200_days = data.Close.rolling(200).mean()


# In[8]:


plt.figure(figsize=(8,6))
plt.plot(ma_100_days, 'r')
plt.plot(ma_200_days,'b')
plt.plot(data.Close,'g')
plt.show()


# In[9]:


data.dropna(inplace=True)


# In[10]:


data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])


# In[11]:


data_train.shape[0]


# In[12]:


data_test.shape[0]


# In[13]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))


# In[14]:


data_train_scale = scaler.fit_transform(data_train)


# In[15]:


x = []
y = []

for i in range(100, data_train_scale.shape[0]):
    x.append(data_train_scale[i-100:i])
    y.append(data_train_scale[i,0])
    


# In[16]:


x, y = np.array(x), np.array(y)


# In[17]:


from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential


# In[18]:


model = Sequential()
model.add(LSTM(units = 50, activation = 'relu', return_sequences = True,
               input_shape = ((x.shape[1],1))))
model.add(Dropout(0.2))

model.add(LSTM(units = 60, activation='relu', return_sequences = True))
model.add(Dropout(0.3))

model.add(LSTM(units = 80, activation = 'relu', return_sequences = True))
model.add(Dropout(0.4))

model.add(LSTM(units = 120, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(units =1))


# In[19]:


model.compile(optimizer = 'adam', loss = 'mean_squared_error')


# In[20]:


model.fit(x,y, epochs = 50, batch_size =32, verbose =1)


# In[21]:


model.summary()


# In[22]:


pas_100_days = data_train.tail(100)


# In[23]:


data_test = pd.concat([pas_100_days, data_test], ignore_index=True)


# In[24]:


data_test_scale  =  scaler.fit_transform(data_test)


# In[25]:


x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])
x, y = np.array(x), np.array(y)


# In[26]:


y_predict = model.predict(x)


# In[27]:


scale =1/scaler.scale_


# In[28]:


y_predict = y_predict*scale


# In[29]:


y = y*scale


# In[30]:


plt.figure(figsize=(10,8))
plt.plot(y_predict, 'r', label = 'Predicted Price')
plt.plot(y, 'g', label = 'Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()


# In[3]:


model.save('StockPP_TataMotors.keras')


# In[ ]:




