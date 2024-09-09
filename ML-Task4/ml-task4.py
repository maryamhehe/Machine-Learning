import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from scipy import stats
from sklearn.metrics import classification_report

dataset= pd.read_csv('C:/Machine-learning/ML-Task4/transaction_anomalies_dataset.csv')
print(dataset.head())

# Check for null values, column info, and descriptive statistics
print("Check Null Values")
null_val = dataset.isnull().sum()
print(null_val)

print("Column Information")
print(dataset.info())

print("Descriptive Statistics of the dataset")
print(dataset.describe())

# #check distribution of transaction amount
plt.figure(figsize=(6,4))
sns.histplot(dataset['Transaction_Amount'])
plt.title('Distribution of Transaction Amount')
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x='Account_Type', y='Transaction_Amount', data=dataset)
plt.title('Transaction Amount by Account Type')
plt.show()

plt.figure(figsize=(6,4))
sns.scatterplot(x='Age', y='Average_Transaction_Amount', hue= 'Account_Type', data=dataset)
sns.regplot(x='Age', y='Average_Transaction_Amount', data=dataset, scatter=False, color='red')
plt.title('Average Transaction Amount vs. Age')
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x='Day_of_Week', data=dataset)
plt.show()

numericData=dataset.select_dtypes(include=['float64','int64'])
corrMatrix= numericData.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corrMatrix, annot=True,fmt=".2f", cmap='viridis')
plt.title("Correlation Matrix")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


#question no 7
dataset['Z_Score'] = stats.zscore(dataset['Transaction_Amount'])
threshold = 3
dataset['Is_Anomaly'] = (dataset['Z_Score'].abs() > threshold)
print(dataset[['Transaction_Amount', 'Z_Score', 'Is_Anomaly']])
plt.figure(figsize=(10,6))
sns.scatterplot(x='Transaction_Amount', y='Average_Transaction_Amount', hue='Is_Anomaly', data=dataset, palette={False: 'blue', True: 'red'})
plt.title('Anomalies in Transaction Amount')
plt.show()

totalAnomalies= dataset['Is_Anomaly'].sum()
totalRecords= len(dataset)
anomalyRatio= totalAnomalies/totalRecords
print(f"Number of anomalies: {totalAnomalies}")
print(f"Total number of records: {totalRecords}")
print(f"Ratio of anomalies: {anomalyRatio:.4f}")

#q 9
features= ['Transaction_Amount','Average_Transaction_Amount','Frequency_of_Transactions']
X= dataset[features]
model= IsolationForest(contamination=0.02, random_state=42)
model.fit(X)
dataset['IF_Prediction']=model.predict(X)
dataset['Bin_Anomaly']=dataset['IF_Prediction'].apply(lambda x:1 if x==-1 else 0)

print(dataset[['Transaction_Amount', 'Average_Transaction_Amount', 'Transaction_Volume', 'Bin_Anomaly']].head())

plt.figure(figsize=(6,4))
sns.scatterplot(x='Transaction_Amount',y='Average_Transaction_Amount',hue='Bin_Anomaly',data=dataset,palette={0:'blue',1:'red'})
plt.title('Anomalies Detected by Isolation Forest')
plt.show()

#q 10
trueLabels= dataset['Is_Anomaly']
predictedLabels=dataset['Bin_Anomaly']
report= classification_report(trueLabels, predictedLabels, target_names=['Normal','Anomaly'])
print(report)

#q 11
transactionAmt=float(input("Enter the value for Transaction Amount: "))
avgTransactionAmt=float(input("Enter the value for Average Transaction Amount: "))
freqTransactionAmt=float(input("Enter the value for Frequency of Transaction Amount: "))

new_data = pd.DataFrame([[transactionAmt, avgTransactionAmt, freqTransactionAmt]], columns=features)
prediction= model.predict(new_data)

if prediction[0]== -1:
    print("Anomaly detected: This transaction is flagged as an anomlay")
else:
    print("This transaction is not flagged as an anomaly")    
