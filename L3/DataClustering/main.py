import numpy as np
import matplotlib.pyplot as plt

# Generates a test sequence of n 2D points
def generate_test_sequence(n):
    return np.random.rand(n, 2)

# Calculate the Euclidean distance between two points x1 and x2
def distance_measure(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# K-Means clustering algorithm
def k_means_clustering(X, n_clusters=3):
    # Initialize centroids randomly
    centroids = X[np.random.choice(X.shape[0], n_clusters, replace=False)]
    # Initialize labels
    labels = np.zeros(X.shape[0])
    # Initialize error
    error = np.inf
    # Loop until convergence
    while True:
        # Calculate distances between each point and each centroid
        distances = np.sqrt(((X - centroids[:, np.newaxis]) ** 2).sum(axis=2))
        # Assign each point to the closest centroid
        new_labels = np.argmin(distances, axis=0)
        # Check if labels have changed
        if (new_labels == labels).all():
            break
        # Update labels
        labels = new_labels
        # Update centroids
        for i in range(n_clusters):
            centroids[i] = X[labels == i].mean(axis=0)
        # Calculate error
        new_error = ((X - centroids[labels]) ** 2).sum()
        # Check if error has changed significantly
        if abs(new_error - error) < 1e-4:
            break
        # Update error
        error = new_error
    return labels

# Hierarchical clustering algorithm
def hierarchical_clustering(X, n_clusters=3):
    # Compute the distance matrix
    dist_matrix = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            dist_matrix[i, j] = np.linalg.norm(X[i] - X[j])
    # Initialize clusters
    clusters = [[i] for i in range(X.shape[0])]
    # Perform agglomerative clustering
    for k in range(X.shape[0] - n_clusters):
        # Find the two closest clusters
        min_dist = float('inf')
        min_i = 0
        min_j = 0
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                # Compute the distance between clusters i and j
                dist = 0
                for a in clusters[i]:
                    for b in clusters[j]:
                        dist += dist_matrix[a, b]
                dist /= len(clusters[i]) * len(clusters[j])
                # Update min_dist, min_i, and min_j
                if dist < min_dist:
                    min_dist = dist
                    min_i = i
                    min_j = j
        # Merge the two closest clusters
        clusters[min_i].extend(clusters[min_j])
        del clusters[min_j]
    # Assign each data point to a cluster
    labels = np.zeros(X.shape[0], dtype=int) # change this line to create an array of type int64 instead of float64
    for i in range(n_clusters):
        for j in clusters[i]:
            labels[j] = i
    return labels

# Calculate the average size of clusters
def avg_cluster_size(X, labels):
    n_clusters = len(np.unique(labels)) # Find the number of unique clusters
    cluster_sizes = np.zeros(n_clusters) # Initializes an array to store the size of each cluster
    for i in range(n_clusters):
        cluster_members = X[labels == i]
        cluster_size = 0
        # Calculate the size by finding the sum of distances between all pairs of data points in that cluster
        for j in range(cluster_members.shape[0]):
            for k in range(j+1, cluster_members.shape[0]):
                cluster_size += distance_measure(cluster_members[j], cluster_members[k])
        # Normalize the cluster size by dividing it by the total number of pairs in that cluster
        cluster_sizes[i] = cluster_size / (cluster_members.shape[0] * (cluster_members.shape[0]-1) / 2)
    # Returns the weighted average of all cluster sizes
    return np.average(cluster_sizes, weights=np.bincount(labels))

# Generate a test sequence of 1000 2D points
X = generate_test_sequence(1000)

# Apply K-Means and agglomerative clustering algorithms to the test sequence
k_means_labels = k_means_clustering(X)
hierarchical_labels = hierarchical_clustering(X)

# Calculate the average size of clusters found by each algorithm
k_means_size = avg_cluster_size(X, k_means_labels)
hierarchical_size = avg_cluster_size(X, hierarchical_labels)

# Print the results of K-Means clustering
print("K-Means Clustering Results:")
print("Number of Clusters:", len(np.unique(k_means_labels)))
print("Average Cluster Size:", k_means_size)
# Print the results of Agglomerative clustering
print("\nAgglomerative Clustering Results:")
print("Number of Clusters:", len(np.unique(hierarchical_labels)))
print("Average Cluster Size:", hierarchical_size)

fig, axs = plt.subplots(1, 2)
# Plot the test sequence with points colored by their K-Means cluster labels
axs[0].scatter(X[:, 0], X[:, 1], c=k_means_labels)
axs[0].set_title('K-Means Clustering')
# Plot the test sequence with points colored by their agglomerative cluster labels
axs[1].scatter(X[:, 0], X[:, 1], c=hierarchical_labels)
axs[1].set_title('Agglomerative Clustering')
plt.show()
