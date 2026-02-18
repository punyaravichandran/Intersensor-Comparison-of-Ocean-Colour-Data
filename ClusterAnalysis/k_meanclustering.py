# -*- coding: utf-8 -*-
"""K-meanClustering.ipynb

@author: Punya  P
"""

#Load necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans

#Load CSV file
Chl_Data = pd.read_csv(r'/content/Chl_March_5_20_wholedataset.csv')
Refl_Data = pd.read_csv(r'/content/Reflectance_March_5_20_WholeData.csv')

#Drop unnecessary coloumns
Refl_Data = Refl_Data.drop(columns=['Unnamed: 0','latitude','longitude'])
#Data scaling
Refl_Data = Refl_Data.multiply(1000)

                                      # For Chlorophyll data #
                                      
# Select the features for clustering
features = ['PaceChl', 'ModisChl']  # Replace with your actual feature names
X = Chl_Data[features]

# ---------- Elbow Method ----------
wcss = []  # Within-cluster sum of squares
max_k = 10  # Maximum number of clusters to try

for i in range(1, max_k + 1):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(5, 5))
plt.plot(range(1, max_k + 1), wcss, marker='o')
#plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of Clusters (k)', fontsize=12)
plt.ylabel('WCSS (Inertia)', fontsize=12)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.grid(True)
plt.show()

#K-means clustering#
# Specify the number of clusters (k)
k = 4
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X)
# Add cluster labels to the dataframe
X['cluster'] = kmeans.labels_
cluster_marker_size = 20 # Change this value to your desired size for cluster markers
centroid_marker_size = 50
# Plot the clusters
plt.figure(figsize=(5, 5))
for i in range(k):
    cluster_data = X[X['cluster'] == i]
    plt.scatter(cluster_data['PaceChl'], cluster_data['ModisChl'], label=f'Cluster {i}')

# Plot centroids
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], s=centroid_marker_size, c='black', marker='X', label='Centroids')

plt.xlabel('Chlorophyll from PACE (mg/m3)',fontsize=12)
plt.ylabel('Chlorophyll from MODIS (mg/m3)',fontsize=12)
plt.title('K-means Clustering of Chlorophyll Values')
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.legend()
plt.show()

                                                  # For Reflectance #
                                                  
# Select the features for clustering
features = ['Rrs_547','MODRrs_547']  # Replace with your actual feature names
X = Refl_Data[features]

# ---------- Elbow Method ----------
wcss = []  # Within-cluster sum of squares
max_k = 10  # Maximum number of clusters to try

for i in range(1, max_k + 1):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(5, 5))
plt.plot(range(1, max_k + 1), wcss, marker='o')
plt.xlabel('Number of Clusters (k)', fontsize=11)
plt.ylabel('WCSS (Inertia)', fontsize=11)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.grid(True)
plt.show()

# Perform K-means clustering - 
#Rrs412
features = ['Rrs_412','MODRrs_412']
X1 = Refl_Data[features].copy()
k = 3
kmeans1 = KMeans(n_clusters=k, random_state=42)
kmeans1.fit(X1)
X1['cluster'] = kmeans1.labels_

#Rrs443
features = ['Rrs_443','MODRrs_443']
X2 = Refl_Data[features].copy()
kmeans2 = KMeans(n_clusters=k, random_state=42)
kmeans2.fit(X2)
X2['cluster'] = kmeans2.labels_

#Rrs469
features = ['Rrs_469','MODRrs_469']
X3 = Refl_Data[features].copy()
kmeans3 = KMeans(n_clusters=k, random_state=42)
kmeans3.fit(X3)
X3['cluster'] = kmeans3.labels_

#Rrs488
features = ['Rrs_488','MODRrs_488']
X4 = Refl_Data[features].copy()
kmeans4 = KMeans(n_clusters=k, random_state=42)
kmeans4.fit(X4)
X4['cluster'] = kmeans4.labels_

#Rrs531
features = ['Rrs_531','MODRrs_531']
X5 = Refl_Data[features].copy()
kmeans5 = KMeans(n_clusters=k, random_state=42)
kmeans5.fit(X5)
X5['cluster'] = kmeans5.labels_

#Rrs547
features = ['Rrs_547','MODRrs_547']
X6 = Refl_Data[features].copy()
kmeans6 = KMeans(n_clusters=k, random_state=42)
kmeans6.fit(X6)
X6['cluster'] = kmeans6.labels_

#Rrs555
features = ['Rrs_555','MODRrs_555']
X7 = Refl_Data[features].copy()
kmeans7 = KMeans(n_clusters=k, random_state=42)
kmeans7.fit(X7)
X7['cluster'] = kmeans7.labels_

#Rrs645
features = ['Rrs_645','MODRrs_645']
X8 = Refl_Data[features].copy()
kmeans8 = KMeans(n_clusters=k, random_state=42)
kmeans8.fit(X8)
X8['cluster'] = kmeans8.labels_

#Rrs667
features = ['Rrs_667','MODRrs_667']
X9 = Refl_Data[features].copy()
kmeans9 = KMeans(n_clusters=k, random_state=42)
kmeans9.fit(X9)
X9['cluster'] = kmeans9.labels_

#Rrs678
features = ['Rrs_678','MODRrs_678']
X10 = Refl_Data[features].copy()
kmeans10 = KMeans(n_clusters=k, random_state=42)
kmeans10.fit(X10)
X10['cluster'] = kmeans10.labels_


# Creates scatterplot for 10 reflectance
# Plot the clusters
fig, axs = plt.subplots(3, 4, figsize=(16, 11))

# Define the marker size for the cluster points and centroids
cluster_marker_size = 20 # Change this value to your desired size for cluster markers
centroid_marker_size = 50 # This is the size for the centroids

#412
for i in range(k):
    cluster_data = X1[X1['cluster'] == i]
    axs[0,0].scatter(cluster_data['Rrs_412'], cluster_data['MODRrs_412'], label=f'Cluster {i}', s=cluster_marker_size)
centroids1 = kmeans1.cluster_centers_
axs[0,0].set_title('Rrs 412') # Set subtitle for this plot

#443
for i in range(k):
    cluster_data = X2[X2['cluster'] == i]
    axs[0,1].scatter(cluster_data['Rrs_443'], cluster_data['MODRrs_443'], label=f'Cluster {i}', s=cluster_marker_size)
centroids2 = kmeans2.cluster_centers_
axs[0,1].set_title('Rrs 443') # Set subtitle for this plot

#469
for i in range(k):
    cluster_data = X3[X3['cluster'] == i]
    axs[0,2].scatter(cluster_data['Rrs_469'], cluster_data['MODRrs_469'], label=f'Cluster {i}', s=cluster_marker_size)
centroids3 = kmeans3.cluster_centers_
axs[0,2].set_title('Rrs 469') # Set subtitle for this plot

#488
for i in range(k):
    cluster_data = X4[X4['cluster'] == i]
    # Corrected column name from 'Rrs_448' to 'Rrs_488'
    axs[0,3].scatter(cluster_data['Rrs_488'], cluster_data['MODRrs_488'], label=f'Cluster {i}', s=cluster_marker_size)
centroids4 = kmeans4.cluster_centers_
axs[0,3].set_title('Rrs 488') # Set subtitle for this plot

#531
for i in range(k):
    cluster_data = X5[X5['cluster'] == i]
    axs[1,0].scatter(cluster_data['Rrs_531'], cluster_data['MODRrs_531'], label=f'Cluster {i}', s=cluster_marker_size)
centroids5 = kmeans5.cluster_centers_
axs[1,0].set_title('Rrs 531') # Set subtitle for this plot

#547
for i in range(k):
    cluster_data = X6[X6['cluster'] == i]
    axs[1,1].scatter(cluster_data['Rrs_547'], cluster_data['MODRrs_547'], label=f'Cluster {i}', s=cluster_marker_size)
centroids6 = kmeans6.cluster_centers_
axs[1,1].set_title('Rrs 547') # Set subtitle for this plot

#555
for i in range(k):
    cluster_data = X7[X7['cluster'] == i]
    axs[1,2].scatter(cluster_data['Rrs_555'], cluster_data['MODRrs_555'], label=f'Cluster {i}', s=cluster_marker_size)
centroids7 = kmeans7.cluster_centers_
axs[1,2].set_title('Rrs 555') # Set subtitle for this plot

#645
for i in range(k):
    cluster_data = X8[X8['cluster'] == i]
    axs[1,3].scatter(cluster_data['Rrs_645'], cluster_data['MODRrs_645'], label=f'Cluster {i}', s=cluster_marker_size)
centroids8 = kmeans8.cluster_centers_
axs[1,3].set_title('Rrs 645') # Set subtitle for this plot

#667
for i in range(k):
    cluster_data = X9[X9['cluster'] == i]
    axs[2,0].scatter(cluster_data['Rrs_667'], cluster_data['MODRrs_667'], label=f'Cluster {i}', s=cluster_marker_size)
centroids9 = kmeans9.cluster_centers_
axs[2,0].set_title('Rrs 667') # Set subtitle for this plot

#678
for i in range(k):
    cluster_data = X10[X10['cluster'] == i]
    axs[2,1].scatter(cluster_data['Rrs_678'], cluster_data['MODRrs_678'], label=f'Cluster {i}', s=cluster_marker_size)
centroids10 = kmeans10.cluster_centers_
axs[2,1].set_title('Rrs 678') # Set subtitle for this plot


# Plot centroids with adjusted size
axs[0,0].scatter(centroids1[:, 0], centroids1[:, 1], s=centroid_marker_size, c='black', marker='X', label='Centroids')
axs[0,1].scatter(centroids2[:, 0], centroids2[:, 1], s=centroid_marker_size, c='black', marker='X', label='Centroids')
axs[0,2].scatter(centroids3[:, 0], centroids3[:, 1], s=centroid_marker_size, c='black', marker='X', label='Centroids')
axs[0,3].scatter(centroids4[:, 0], centroids4[:, 1], s=centroid_marker_size, c='black', marker='X', label='Centroids')
axs[1,0].scatter(centroids5[:, 0], centroids5[:, 1], s=centroid_marker_size, c='black', marker='X', label='Centroids')
axs[1,1].scatter(centroids6[:, 0], centroids6[:, 1], s=centroid_marker_size, c='black', marker='X', label='Centroids')
axs[1,2].scatter(centroids7[:, 0], centroids7[:, 1], s=centroid_marker_size, c='black', marker='X', label='Centroids')
axs[1,3].scatter(centroids8[:, 0], centroids8[:, 1], s=centroid_marker_size, c='black', marker='X', label='Centroids')
axs[2,0].scatter(centroids9[:, 0], centroids9[:, 1], s=centroid_marker_size, c='black', marker='X', label='Centroids')
axs[2,1].scatter(centroids10[:, 0], centroids10[:, 1], s=centroid_marker_size, c='black', marker='X', label='Centroids')


# Set labels for each subplot
axs[0,0].set_ylabel('Rrs MODIS', fontsize=11)
axs[1,0].set_ylabel('Rrs MODIS', fontsize=11)
axs[2,0].set_xlabel('Rrs PACE')
axs[2,0].set_ylabel('Rrs MODIS', fontsize=11)
axs[2,1].set_xlabel('Rrs PACE')


# Add a single legend for all subplots outside the plot area
handles, labels = axs[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.2, 0.5), fontsize=20)

# Adjust layout to prevent labels from overlapping
plt.tight_layout(rect=[0, 0, 1, 1])
plt.show()





