import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sb

user_behavior= pd.read_csv('C:/Machine-learning/ML-Task2/userbehaviour.csv')
print(user_behavior.head())

print("Null values in the data: ")
null_val=user_behavior.isnull().sum()
print(null_val)

print("Column info of the data: ")
print(user_behavior.info())

print("Descriptive Statistics of the Data")
print(user_behavior.describe())

#question 2
highest_screen_time= user_behavior['Average Screen Time'].max()
average_screen_time= user_behavior['Average Screen Time'].mean()
lowest_screen_time= user_behavior['Average Screen Time'].min()

print(f"Highest Screen Time: {highest_screen_time}")
print(f"Average Screen Time: {average_screen_time}")
print(f"Lowest Screen Time: {lowest_screen_time}")

#question no 3
highest_spent_time= user_behavior['Average Spent on App (INR)'].max()
average_spent_time= user_behavior['Average Spent on App (INR)'].mean()
lowest_spent_time= user_behavior['Average Spent on App (INR)'].min()

print(f"Highest Spent Time: {highest_spent_time}")
print(f"Average Spent Time: {average_spent_time}")
print(f"Lowest Spent Time: {lowest_spent_time}")

#question no 4
activeUsers= user_behavior[user_behavior['Status']=='Installed']
inactiveUsers= user_behavior[user_behavior['Status']=='Uninstalled']

activeUserSpent= activeUsers['Average Spent on App (INR)'].mean()
activeUserScreenTime= activeUsers['Average Screen Time'].mean()

inactiveUserSpent= inactiveUsers['Average Spent on App (INR)'].mean()
inactiveUserScreenTime= inactiveUsers['Average Screen Time'].mean()

print("Active Users")
print(f"Active users' Average spent time (INR): {activeUserSpent}")
print(f"Active users' Average screen time: {activeUserScreenTime}")
print("Inactive Users")
print(f"Inactive users' Average spent time: {inactiveUserSpent}")
print(f"Inactive users' Average screen time: {inactiveUserScreenTime}")

# #Scatter Plot
plt.figure(figsize=(12,6))
sb.scatterplot(x='Average Screen Time', y='Average Spent on App (INR)', data=activeUsers, label='Active Users')
sb.scatterplot(x='Average Screen Time', y='Average Spent on App (INR)', data=inactiveUsers, label= 'Inactive Users')
plt.title('Relationship between Screen Time and Spending')
plt.xlabel('Average Screen Time')
plt.ylabel('Average Spent on App (INR)')
plt.legend()
plt.show()

#Observation
# Active Users:
# They have higher average spending and higher average screen time.
# This indicates that users who are more engaged (spend more time) tend to spend more money on the app.

# Inactive Users:
# They have lower average spending and lower average screen time.
# This suggests that users who are less engaged (spend less time) are less likely to spend money and more 
# likely to uninstall the app.

#question no 5
plt.figure(figsize=(12,6))
sb.scatterplot(x='Average Screen Time', y='Ratings', hue='Status', data=user_behavior, palette={'Installed':'blue','Uninstalled':'red'})
plt.title('Relationship between User Ratings and Average Screen Time')
plt.xlabel('Average Screen Time')
plt.ylabel('Ratings')
plt.show()
#Observation 
# For users who have uninstalled the app, screen time is generally low (mostly below 10). 
# This suggests that users who don't spend much time on the app are more likely to uninstall
# it and give lower ratings Users who have uninstalled the app (red dots) tend to have given lower ratings.
# Users who have the app installed (blue dots) have a wider range of ratings, 
# with many giving high ratings (6 to 10) Active users (with the app installed) have a wide range of screen times, 
# indicating that both heavy and light users are likely to retain the app.

#question no 6
features= user_behavior[['Average Screen Time','Average Spent on App (INR)','Ratings','New Password Request','Last Visited Minutes']]

scaler= StandardScaler()
scalerFeatures= scaler.fit_transform(features)

inertias=[]
K= range(1,11)

for k in K:
    kmeans= KMeans(n_clusters=k, random_state=40)
    kmeans.fit(features)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(12,6))
plt.plot(K, inertias, 'bo-')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.show()

clusters=2
kmeans = KMeans(n_clusters=clusters, random_state=42)
user_behavior['Cluster'] = kmeans.fit_predict(features)

print(user_behavior.columns)
print(user_behavior['Cluster'].head())

numeric_columns = user_behavior.select_dtypes(include='number').columns
print(user_behavior.groupby('Cluster')[numeric_columns].mean())

cluster_status_counts = user_behavior.groupby(['Cluster', 'Status']).size().unstack(fill_value=0)
print("Number of users in each cluster by status:")
print(cluster_status_counts)

plt.figure(figsize=(12, 8))

cluster_colors = {0: 'blue', 1: 'red', 2: 'green'}
cluster_labels = {0: 'Retained', 1: 'Needs Attention', 2: 'Churn'}


for cluster in range(clusters):
    clustered_data = user_behavior[user_behavior['Cluster'] == cluster]
    plt.scatter(clustered_data['Last Visited Minutes'],
                clustered_data['Average Spent on App (INR)'],
                color=cluster_colors[cluster],
                label=cluster_labels[cluster])

plt.title('User Segments Based on Clustering')
plt.xlabel('Last Visited Minutes')
plt.ylabel('Average Spent on App (INR)')
plt.legend()
plt.show()

#question no 8
#SUMMARY OF THE WORK
# Loaded and explored the dataset.
# Performed basic calculations and visualizations.
# Applied K-Means clustering to segment users.
# Visualized the clusters to identify patterns in user behavior.
# The final plot clearly shows three distinct user segments: 'Retained', 'Needs Attention', 
# and 'Churn', differentiated by their last visited minutes and average spending on the app.