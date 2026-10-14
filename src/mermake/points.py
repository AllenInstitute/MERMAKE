import numpy as np
from scipy.spatial import KDTree


def find_two_means(values):
	from sklearn.cluster import KMeans
	values = np.abs(values).reshape(-1, 1)  # Reshape for clustering
	kmeans = KMeans(n_clusters=3, n_init="auto").fit(values)
	cluster_centers = kmeans.cluster_centers_
	return sorted(cluster_centers.flatten())[:2]

def estimate_step_size(points):
	points = np.array(points)
	# Build a KD-tree for efficient nearest neighbor search
	tree = KDTree(points)
	# Find the distance to the nearest neighbor for each point
	distances, _ = tree.query(points, k=2)  # k=2 because first result is the point itself
	nearest_dists = distances[:, 1]  # Extract nearest neighbor distances (skip self-distance)
	# Use the median to ignore outliers (or mode if step size is very regular)
	step_size = np.median(nearest_dists)  # More robust than mean
	return step_size

def points_to_coords(points):
	'convert xy point locations to integer grid coordinates'
	points = np.array(points)
	points -= np.min(points, axis=0)
	#_,mean = find_two_means(shifts)
	mean = estimate_step_size(points)
	coords = np.round(points / mean).astype(int)
	return coords
