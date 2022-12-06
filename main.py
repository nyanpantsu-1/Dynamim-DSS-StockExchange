import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.layers import Dense , Dropout , LSTM
from keras.models import Sequential

#load data
company = 'GOOG'

data=web.DataReader(company,'yahoo',start='2021-01-01',end='2022-06-01')

train_data=data


scaler= MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(train_data['Close'].values.reshape(-1,1))

prediction_days= 60

x_train=[]
y_train=[]

for x in range(prediction_days,len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x,0])
    y_train.append(scaled_data[x,0])
    

x_train,y_train=np.array(x_train),np.array(y_train)
x_train=np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))


#building model
model= Sequential()

model.add(LSTM(units=50, return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train,y_train,epochs=50,batch_size=16)


'''testing model'''
test_data=web.DataReader(company,'yahoo',start='2022-06-02',end='2022-12-01')
actual_prices=test_data['Close'].values
total_dataset=pd.concat((train_data['Close'],test_data['Close']),axis=0)

model_inputs=total_dataset[len(total_dataset)-len(test_data)-prediction_days:].values
model_inputs=model_inputs.reshape(-1,1)
model_inputs=scaler.transform(model_inputs)

x_test=[]

for x in range(prediction_days,len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x,0])

x_test=np.array(x_test)
x_test=np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

predicted_prices=model.predict(x_test)
predicted_prices=scaler.inverse_transform(predicted_prices)

#plot 
plt.plot(actual_prices,color="black",label=f"Actual {company} price")
plt.plot(predicted_prices,color="green",label=f"Predicted {company} price")
plt.title(f"{company} Stock Price")
plt.xlabel('Time')
plt.ylabel(f"{company} Stock Price")
plt.legend()
plt.show()

#predict 

real_data=[model_inputs[len(model_inputs)+1-prediction_days:len(model_inputs+1),0]]
real_data=np.array(real_data)
real_data=np.reshape(real_data,(real_data.shape[0],real_data.shape[1],1))

prediction=model.predict(real_data)
prediction= scaler.inverse_transform(prediction)
print(f"predicted next day price :{prediction}")


