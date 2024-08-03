import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
# Load the dataset
user_prof = pd.read_csv('C:/Machine-learning/ML-Task3/user_profiles_for_ads.csv')

# Check for null values, column info, and descriptive statistics
print("Check Null Values")
null_val = user_prof.isnull().sum()
print(null_val)

print("Column Information")
print(user_prof.info())

print("Descriptive Statistics of the user_prof")
print(user_prof.describe())

fig, axis = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Distribution of Key Demographic Variables")
sns.histplot(user_prof, x='Gender', hue='Gender', ax=axis[0,0])
sns.histplot(user_prof, x='Education Level', hue='Education Level', ax=axis[0,1])
sns.histplot(user_prof, x='Age', hue='Age', ax=axis[1,0])
sns.histplot(user_prof, x='Income Level', hue='Income Level', ax=axis[1,1])
fig.tight_layout()
plt.show()


sns.histplot(user_prof, x='Device Usage', hue='Device Usage')
plt.show()  

# #question no 4
fig2, axis2= plt.subplots(3,2,figsize=(10,8))
fig.suptitle("User Online Behavior and AD Interaction Metrics")
sns.histplot(user_prof, x='Time Spent Online (hrs/weekday)', color='blue',ax=axis2[0,0], kde=True)
sns.histplot(user_prof, x='Time Spent Online (hrs/weekend)', color='yellow', ax=axis2[0,1], kde=True)
sns.histplot(user_prof, x='Likes and Reactions', color='green', ax=axis2[1,0], kde=True)
sns.histplot(user_prof, x='Click-Through Rates (CTR)', color='red', ax=axis2[1,1], kde=True)
sns.histplot(user_prof, x='Conversion Rates', color='purple', ax=axis2[2,0],kde=True)
sns.histplot(user_prof, x='Ad Interaction Time (sec)', color='brown', ax=axis2[2,1],kde=True)
fig2.tight_layout()
plt.show()

# question no 5
top_interests= user_prof['Top Interests'].value_counts().head(10)
plt.figure(figsize=(10,8))
sns.barplot(x=top_interests.values, y=top_interests.index, palette='coolwarm')
plt.title('Top 10 User Interests')
plt.xlabel('Frequency')
plt.ylabel('Interests')
plt.show()

for interest in top_interests.index:
    user_prof['Top 10 Interests']= user_prof['Top Interests'].apply(lambda x: 1 if interest in x else 0)
print(user_prof.head(16))
#question no 6

selectedFeatures= user_prof[['Age','Gender', 'Income Level', 'Education Level', 'Time Spent Online (hrs/weekday)', 
                        'Time Spent Online (hrs/weekend)', 'Likes and Reactions', 
                        'Click-Through Rates (CTR)','Top 10 Interests']]

print(selectedFeatures.head())

categoricalFeatures = ['Age', 'Gender', 'Income Level', 'Education Level']
numericalFeatures = [col for col in selectedFeatures.columns if col not in categoricalFeatures]

column_transformer = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numericalFeatures),
        ('cat', OneHotEncoder(), categoricalFeatures)
    ]
)

scalerFeatures = column_transformer.fit_transform(selectedFeatures)
   

kmeans= KMeans(n_clusters=5, random_state=100)
user_prof['Clusters']=kmeans.fit_predict(scalerFeatures)

#question no 7

clusterMean= user_prof.groupby('Clusters')[numericalFeatures].mean()
clusterMode= user_prof.groupby('Clusters')[categoricalFeatures].agg(lambda x: x.mode().iloc[0])
print("Cluster Means (Numerical Features):")
print(clusterMean)
print("Cluster Modes (Categorical Features):")
print(clusterMode)

#question no 8

clusterNames = {
    0: "Weekend Warriors",
    1: "Engaged Professionals",
    2: "Low-Key Users",
    3: "Active Explorers",
    4: "Budget Browsers"
}
user_prof['Cluster Name'] = user_prof['Clusters'].map(clusterNames)

# Normalize the data for radar chart
clusterMean_normalized = (clusterMean - clusterMean.min()) / (clusterMean.max() - clusterMean.min())
print("Normalized Cluster Means:")
print(clusterMean_normalized)

#question no 9
# Function to create a combined radar chart
def create_combined_radar_chart(cluster_data, labels, categories, colors):
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for i, (cluster, color) in enumerate(zip(cluster_data.index, colors)):
        data = cluster_data.loc[cluster].values
        data = np.concatenate((data, [data[0]]))
        
        ax.fill(angles, data, color=color, alpha=0.25)
        ax.plot(angles, data, color=color, linewidth=2, label=labels[cluster])

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    plt.legend(loc='upper right', bbox_to_anchor=(1.6, 1.1))
    plt.title("User Segments Profile")
    plt.show()

# Categories (features) for the radar chart
categories = list(clusterMean_normalized.columns)

colors = ['blue', 'orange', 'purple', 'green', 'red']

# Create a combined radar chart for all clusters
create_combined_radar_chart(clusterMean_normalized, clusterNames, categories, colors)


#SUMMARY OF THE ABOVE WORK

# The analysis segments users into five distinct profiles: 
# Weekend Warriors, Engaged Professionals, Low-Key Users, Active Explorers, 
# and Budget Browsers. Each segment displays unique online behaviors and 
# engagement levels, which are crucial for tailoring marketing strategies.

# The radar chart effectively visualizes these differences, showing how each 
# user group interacts with various aspects of online content. This analysis 
# can help businesses target their marketing efforts more effectively based 
# on the identified user segments.