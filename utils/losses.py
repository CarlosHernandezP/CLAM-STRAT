import torch
from lifelines import CoxPHFitter
import numpy as np
import pandas as pd

def silhouette_loss(data, assignments):
    # Compute pairwise distances
    pairwise_distances = torch.cdist(data, data)

    # Expand assignments for broadcasting
    assignments_expanded = assignments.unsqueeze(1)
    # Intra-cluster distances
    intra_cluster_mask = torch.matmul(assignments_expanded, assignments.unsqueeze(2))
    intra_cluster_distances = pairwise_distances * intra_cluster_mask
    sum_intra_cluster_distances = torch.sum(intra_cluster_distances, dim=2)
    count_intra_cluster_distances = torch.sum(intra_cluster_mask, dim=2)
    average_intra_cluster_distance = sum_intra_cluster_distances / count_intra_cluster_distances

    # Inter-cluster distances
    inter_cluster_mask = 1 - intra_cluster_mask
    max_distance = torch.max(pairwise_distances)
    inter_cluster_distances = pairwise_distances * inter_cluster_mask + max_distance * (1 - inter_cluster_mask)
    min_inter_cluster_distance, _ = torch.min(inter_cluster_distances, dim=2)

    # Silhouette scores
    silhouette_scores = (min_inter_cluster_distance - average_intra_cluster_distance) / torch.max(min_inter_cluster_distance, average_intra_cluster_distance)
    silhouette_scores[torch.isnan(silhouette_scores)] = 0

    # Average silhouette score for the batch
    average_silhouette_score = torch.mean(silhouette_scores)
    loss = (1 - average_silhouette_score) / 2
    return loss  # Negative as we want to maximize the silhouette score


def k_means_loss(data, centroids):
    """
    Computes the K-means loss.
    
    Args:
    data (Tensor): The data points (batch_size, features).
    centroids (Tensor): The cluster centroids (num_clusters, features).
    
    Returns:
    Tensor: The K-means loss.
    """
    # Expand data and centroids to calculate distances
    data_expanded = data.unsqueeze(1)  # Shape: (batch_size, 1, features)
    centroids_expanded = centroids.unsqueeze(0)  # Shape: (1, num_clusters, features)

    # Calculate squared L2 distances
    distances = torch.sum((data_expanded - centroids_expanded) ** 2, dim=2)

    # Minimum distance to a centroid for each data point
    min_distances = torch.min(distances, dim=1).values

    # K-means loss is the mean of these minimum distances
    loss = torch.mean(min_distances)

    return loss

def compute_losses(aggregated_data: torch.Tensor, 
                   centroids: torch.Tensor, 
                   assignments: torch.Tensor, 
                   k_means_loss_weight: float, 
                   silhouette_loss_weight: float) -> tuple:
    """
    Computes the total loss as a weighted sum of K-means and Silhouette losses.

    Args:
    aggregated_data (torch.Tensor): The aggregated data points.
    centroids (torch.Tensor): The cluster centroids.
    assignments (torch.Tensor): Soft cluster assignments for each data point.
    k_means_loss_weight (float): Weight for the K-means loss.
    silhouette_loss_weight (float): Weight for the Silhouette loss.

    Returns:
    tuple: Total loss, K-means loss, and Silhouette loss.
    """
    # Compute K-means and Silhouette losses
    loss_k_means = k_means_loss(aggregated_data, centroids)
    loss_silhouette = silhouette_loss(aggregated_data, assignments)

    # Compute total weighted loss
    total_loss = k_means_loss_weight * loss_k_means + silhouette_loss_weight * loss_silhouette

    return total_loss, loss_k_means, loss_silhouette

def compute_clustered_nll(cluster_probabilities, survival_tensor):
    """
    Compute the negative log-likelihood for survival data based on cluster assignments.

    Parameters:
    cluster_probabilities (Tensor): Tensor of cluster probabilities for each instance.
    survival_tensor (Tensor): Tensor with survival data, each element is a tuple (event, time).

    Returns:
    float: Aggregated negative log-likelihood.
    """
    # Get the cluster assignments
    cluster_assignments = torch.argmax(cluster_probabilities, dim=1)

    # Convert tensors to numpy arrays if they are not already
    if isinstance(cluster_assignments, torch.Tensor):
        cluster_assignments = cluster_assignments.detach().cpu().numpy()
    if isinstance(survival_tensor, torch.Tensor):
        survival_tensor = survival_tensor.detach().cpu().numpy()

    # Convert survival data to DataFrame
    survival_data = pd.DataFrame(survival_tensor, columns=['event', 'time'])

    unique_clusters = np.unique(cluster_assignments)
    total_nll = 0
    
    for cluster in unique_clusters:
        # Extract data for the current cluster
        cluster_data = survival_data[cluster_assignments == cluster]

        if not cluster_data.empty:
            # Fit Cox model
            cph = CoxPHFitter()
            cph.fit(cluster_data, duration_col='time', event_col='event')
            
            # Compute negative log-likelihood for this cluster
            cluster_nll = -cph.log_likelihood_
            total_nll += cluster_nll

    return total_nll
