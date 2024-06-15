import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('C:/Users/mukul/Downloads/Mall_Customers/Mall_Customers.csv')

# Display the first few rows of the dataset 
print(data.head())

# Selecting annual income and spending score for clustering
X = data.iloc[:, [3, 4]].values

# Feature scaling 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Elbow Method to find the optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plotting the Elbow Method graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Applying K-means to the dataset
kmeans = KMeans(n_clusters=37, init='k-means++', random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

# Adding cluster labels to the original dataset
data['Cluster'] = cluster_labels

# Analyzing the clusters
cluster_centers = kmeans.cluster_centers_
print("Cluster Centers:")
print(scaler.inverse_transform(cluster_centers))  # inverse transform to interpret back to original scale

# Count of customers in each cluster
print("\nNumber of customers in each cluster:")
print(data['Cluster'].value_counts())


# Visualizing
plt.figure(figsize=(12, 8))

plt.scatter(X_scaled[cluster_labels == 0, 0], X_scaled[cluster_labels == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X_scaled[cluster_labels == 1, 0], X_scaled[cluster_labels == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X_scaled[cluster_labels == 2, 0], X_scaled[cluster_labels == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(X_scaled[cluster_labels == 3, 0], X_scaled[cluster_labels == 3, 1], s=100, c='cyan', label='Cluster 4')
plt.scatter(X_scaled[cluster_labels == 4, 0], X_scaled[cluster_labels == 4, 1], s=100, c='magenta', label='Cluster 5')

plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=300, c='yellow', label='Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

