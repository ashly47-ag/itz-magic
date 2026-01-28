import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

df = yf.download("AAPL", start="2010-01-01", end="2024-01-01")
# print(df.tail())
df = df.reset_index(drop=True)
# df = df.reset_index()
# print(df.head())
# df = df.drop(columns=['Date'])
# print(df.head())
 
ma100 = df['Close'].rolling(100).mean()
ma200 = df['Close'].rolling(200).mean()
plt.figure(figsize=(12,6))
plt.plot(df['Close'], label='Close')
plt.plot(ma100, label='MA 100')
plt.plot(ma200, label='MA 200')
plt.legend()
# plt.show()

data_training = pd.DataFrame(df['Close'].iloc[:int(len(df)*0.70)])
data_testing  = pd.DataFrame(df['Close'].iloc[int(len(df)*0.70):])
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)
x_train = []
y_train = []
for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i, 0])  
    y_train.append(data_training_array[i, 0])      

x_train, y_train = np.array(x_train), np.array(y_train)

# reshape for LSTM: (samples, timesteps, features)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# print(x_train.shape)
# print(y_train.shape)

from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

model = Sequential()
model.add(LSTM(units = 50, activation = 'relu', return_sequences = True, input_shape = (x_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units = 60, activation = 'relu', return_sequences = True))
model.add(Dropout(0.3))

model.add(LSTM(units = 80, activation = 'relu', return_sequences = True))
model.add(Dropout(0.4))

model.add(LSTM(units = 120, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(units =1))
print(model.summary())

model.compile(optimizer='adam', loss = 'mean_squared_error')
model.fit(x_train, y_train, epochs = 50)
model.save('keras_model.h5')

past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)
input_data = scaler.transform(final_df)
x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i, 0])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

y_predicted = model.predict(x_test)
scale_factor = 1 / scaler.scale_[0]

y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

# last_100_days = final_df.tail(100)
# last_100_days_scaled = scaler.transform(last_100_days)

# x_future = []
# x_future.append(last_100_days_scaled[:, 0])
# x_future = np.array(x_future)
# x_future = np.reshape(x_future, (x_future.shape[0], x_future.shape[1], 1))

# future_predictions = []

# for i in range(7):
#     pred = model.predict(x_future, verbose=0)
#     future_predictions.append(pred[0, 0])

#     x_future = np.roll(x_future, -1, axis=1)
#     x_future[0, -1, 0] = pred
# future_predictions = np.array(future_predictions)
# future_predictions = future_predictions * (1 / scaler.scale_[0])
# plt.figure(figsize=(12,6))
# plt.plot(df['Close'].values, label='Historical Price')
# plt.plot(
#     range(len(df), len(df)+7),
#     future_predictions,
#     'r--',
#     label='Next 7 Days Prediction'
# )
# plt.xlabel('Time')
# plt.ylabel('Price')
# plt.legend()
# plt.show()
