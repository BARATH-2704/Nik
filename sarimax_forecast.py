import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv('data/preprocessed_walmart_dataset.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df[(df['Store'] == 1) & (df['Dept'] == 1)].sort_values('Date')
df.set_index('Date', inplace=True)
df.index = pd.DatetimeIndex(df.index).to_period('W').to_timestamp()

y = df['Weekly_Sales']
X = df[['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'IsHoliday']]

split_index = int(len(y) * 0.8)
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]

model = SARIMAX(y_train,
                exog=X_train,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 52),
                enforce_stationarity=False,
                enforce_invertibility=False)
results = model.fit(disp=False)

pred = results.get_prediction(start=len(y_train), end=len(y_train) + len(y_test) - 1, exog=X_test)
forecast = pred.predicted_mean
forecast.index = y_test.index
conf_int = pred.conf_int()

mae = mean_absolute_error(y_test, forecast)
rmse = np.sqrt(mean_squared_error(y_test, forecast))

print("\n✅ SARIMAX Forecast Complete")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}\n")

os.makedirs("output", exist_ok=True)
pd.DataFrame({
    'Date': y_test.index,
    'Actual_Sales': y_test.values,
    'Predicted_Sales': forecast.values
}).to_csv('output/sarimax_forecast_output.csv', index=False)

plt.figure(figsize=(12, 6))
plt.plot(y_train.index, y_train, label='Train')
plt.plot(y_test.index, y_test, label='Actual', color='blue')
plt.plot(forecast.index, forecast, label='Forecast', color='red')
plt.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3)
plt.title('SARIMAX Forecast (Store 1, Dept 1)')
plt.xlabel('Date')
plt.ylabel('Weekly Sales')
plt.legend()
plt.tight_layout()
plt.savefig('output/sarimax_forecast_plot.png')
plt.show()
