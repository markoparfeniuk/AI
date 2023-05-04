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

# ART1 clustering algorithm
def art1_clustering(X, rho=0.5, n_clusters=3):
    # Initialize weights
    W = np.random.rand(n_clusters, X.shape[1])
    # Initialize clusters
    clusters = np.zeros(X.shape[0], dtype=int)
    # Initialize number of iterations
    n_iter = 0
    # Repeat until convergence
    while True:
        n_iter += 1
        for i in range(X.shape[0]):
            x = X[i]
            # Compute activations
            a = np.dot(W, x) / (rho + np.sum(W, axis=1))
            # Find the winning cluster
            j = np.argmax(a)
            # Update the weights of the winning cluster
            W[j] = rho * W[j] + (1 - rho) * x
            # Assign the pattern to the winning cluster
            clusters[i] = j
        if n_iter > 500:
            break
    return clusters

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

# Apply K-Means and ART1 clustering algorithms to the test sequence
k_means_labels = k_means_clustering(X)
art1_labels = art1_clustering(X)

# Calculate the average size of clusters found by each algorithm
k_means_size = avg_cluster_size(X, k_means_labels)
art1_size = avg_cluster_size(X, art1_labels)

# Print the results of K-Means clustering
print("K-Means Clustering Results:")
print("Number of Clusters:", len(np.unique(k_means_labels)))
print("Average Cluster Size:", k_means_size)
# Print the results of ART1 clustering
print("\nART1 Clustering Results:")
print("Number of Clusters:", len(np.unique(art1_labels)))
print("Average Cluster Size:", art1_size)

fig, axs = plt.subplots(1, 2)
# Plot the test sequence with points colored by their K-Means cluster labels
axs[0].scatter(X[:, 0], X[:, 1], c=k_means_labels)
axs[0].set_title('K-Means Clustering')
# Plot the test sequence with points colored by their ART1 cluster labels
axs[1].scatter(X[:, 0], X[:, 1], c=art1_labels)
axs[1].set_title('ART1 Clustering')
plt.show()