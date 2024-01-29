import numpy as np
from lifelines import NelsonAalenFitter
from lifelines.utils import concordance_index

def fit_nelson_aalen_clusters(train_data, val_data):
    """
    Fit Nelson-Aalen estimators for each cluster in training data
    and apply to validation data.

    Parameters:
    train_data (dict): A dictionary with 'clusters', 'times', and 'events' keys for training data.
    val_data (dict): A dictionary with 'clusters', 'times', and 'events' keys for validation data.

    Returns:
    dict: A dictionary with c-index for training and validation data.
    """

    # Unpack training data
    train_clusters = train_data['clusters']
    train_times = train_data['times']
    train_events = train_data['events']

    # Unpack validation data
    val_clusters = val_data['clusters']
    val_times = val_data['times']
    val_events = val_data['events']

    # Fit Nelson-Aalen for each cluster in training data
    naf_estimators = {}
    for cluster in np.unique(train_clusters):
        naf = NelsonAalenFitter()
        idx = train_clusters == cluster
        naf.fit(train_times[idx], event_observed=train_events[idx])
        naf_estimators[cluster] = naf

    # Function to get cumulative hazard for each individual
    def get_cumulative_hazard(clusters, estimators):
        hazards = [estimators[cluster].cumulative_hazard_.values[-1, 0] for cluster in clusters]
        return np.array(hazards)

    # Calculate cumulative hazards for training and validation data
    train_hazards = get_cumulative_hazard(train_clusters, naf_estimators)
    val_hazards = get_cumulative_hazard(val_clusters, naf_estimators)

    # Calculate c-index for training and validation data
    c_index_train = concordance_index(train_times, -train_hazards, train_events)
    c_index_val = concordance_index(val_times, -val_hazards, val_events)

    print(f'Num clusters: {len(naf_estimators)}')

    return {'c_index_train': c_index_train, 'c_index_val': c_index_val}


if __name__ == '__main__':
    # Example usage
    train_data = {
        'clusters': np.array([0, 1, 0, 1]),  # Cluster assignments
        'times': np.array([5, 10, 15, 20]),  # Time-to-event
        'events': np.array([1, 0, 1, 1])     # Event indicator
    }

    val_data = {
        'clusters': np.array([1, 0, 1]),     # Cluster assignments
        'times': np.array([7, 12, 18]),      # Time-to-event
        'events': np.array([1, 1, 0])        # Event indicator
    }

    results = fit_nelson_aalen_clusters(train_data, val_data)
    print(results)

