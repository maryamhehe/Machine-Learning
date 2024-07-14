import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
#Question no 1

df=pd.read_csv('Instagram-Reach.csv')
print(df.head())

#check null values in data
print("If there are any null values in the Data")
null_val= df.isnull().sum()
print(null_val)

#to get column info
print("Column Information of the Data")
print(df.info())

#to get dercriptive statistics of the data
print("Descriptive Stats of the Data")
print(df.describe())

# Question no 2
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df.sort_index(inplace=True)
print(df.dtypes)

# print(df.head())

#Question no 3
y = df['Instagram reach']
plt.xlabel('Date', fontsize=18)
plt.ylabel('Instagram reach', fontsize=18)
plt.plot(df.index, y)
plt.show()


#Question no 4
#bar chart 
plt.bar(df.index,y)
plt.show()

#question no 5
#box plot 
fig= plt.figure(figsize=(10,8))
plt.boxplot(df['Instagram reach'], showmeans=True)
plt.xlabel('Instagram Reach')
plt.ylabel('Values')
plt.show()

#question no 6
df['Day']= df.index.day_name()
print(df.head())
#question no 7
days_order=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
stats_by_day= df.groupby('Day')['Instagram reach'].agg(['mean','median','std']).reindex(days_order)

print("Mean of Instagram Reach by Day of the Week:\n",stats_by_day['mean'])
print("Median of Instagram Reach by Day of the Week:\n",stats_by_day['median'])
print("Std of Instagram Reach by Day of the Week:\n",stats_by_day['std'])

day_x=df['Day']
Insta_y=df['Instagram reach']
plt.bar(day_x,Insta_y)
plt.show()


result= seasonal_decompose(df['Instagram reach'], model='multiplicative',period=30)
plt.plot(result.trend)
plt.title("Trend")
plt.show()
plt.plot(result.seasonal)
plt.title("Seasonal")
plt.show()


value= df['Instagram reach']
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
plot_acf(value, ax=ax1, lags=10)
ax1.set_title('Autocorrelation Function (ACF)')
ax1.set_xlabel('Lag')
ax1.set_ylabel('Autocorrelation')

plot_pacf(value, ax=ax2, lags=50)
ax2.set_title('Partial Autocorrelation Function (PACF)')
ax2.set_xlabel('Lag')
ax2.set_ylabel('Partial Autocorrelation')
plt.show()

p=1 
q=1
d=1
P=0
Q=0
D=1
S=7

model= SARIMAX(df['Instagram reach'],order= (p,d,q), seasonal_order=(P,D,Q,S))
results=model.fit()

steps = 30
forecast = results.get_forecast(steps=steps)
last_date = df.index[-1]
forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq='D')
forecast_mean = forecast.predicted_mean
forecast_conf_int = forecast.conf_int()

# Plot forecast
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Instagram reach'], label='Observed')
plt.plot(forecast_index, forecast_mean, label='Forecast', color='red')
plt.fill_between(forecast_index, forecast_conf_int.iloc[:, 0], forecast_conf_int.iloc[:, 1], color='pink', alpha=0.3)

plt.title('Sarima forecast')
plt.xlabel('Date')
plt.ylabel('Instagram reach')
plt.legend()
plt.show()

print(results.summary())