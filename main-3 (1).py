import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Generate a random dataset
np.random.seed(42)
date_rng = pd.date_range(start='2020-01-01', end='2023-12-31', freq='ME')  # Monthly frequency
data = np.random.poisson(lam=200, size=len(date_rng)) + np.sin(np.linspace(0, 3 * np.pi, len(date_rng))) * 50
df = pd.DataFrame(data, index=date_rng, columns=['Inventory'])

# Fit the SARIMA model
model = SARIMAX(df['Inventory'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results = model.fit()

# Forecasting the next 12 months
forecast = results.get_forecast(steps=12)
forecast_index = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=12, freq='ME')  # Forecast index
forecast_values = forecast.predicted_mean

# Plot the historical data and forecast
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Inventory'], label='Historical Inventory', color='blue')
plt.plot(forecast_index, forecast_values, label='Forecasted Inventory', color='orange')
plt.fill_between(forecast_index, 
								 forecast.conf_int()['lower Inventory'], 
								 forecast.conf_int()['upper Inventory'], 
								 color='gray', alpha=0.2, label='Confidence Interval')
plt.title('Inventory Forecast')
plt.xlabel('Date')
plt.ylabel('Inventory Level')
plt.legend()
plt.show()