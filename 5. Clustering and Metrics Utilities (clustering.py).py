# utils/clustering.py
import numpy as np
try:
    import hdbscan
except ImportError:
    hdbscan = None

def cluster_embeddings(embeddings, min_cluster_size):
    """
    Cluster the given embedding vectors using HDBSCAN.
    :param embeddings: numpy array of shape (N, embed_dim)
    :param min_cluster_size: Minimum cluster size parameter for HDBSCAN.
    :return: An array of cluster labels for each embedding (length N).
    """
    if hdbscan is None:
        raise ImportError("Please install the hdbscan library for clustering.")
    # Initialize HDBSCAN clusterer
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = clusterer.fit_predict(embeddings)
    return labels

# utils/metrics.py
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, v_measure_score

def evaluate_clustering(true_labels, pred_labels):
    """
    Compute clustering metrics (ARI, AMI, V-measure) comparing predicted labels to true labels.
    :param true_labels: array-like of shape (N,) with ground-truth emitter IDs for each pulse.
    :param pred_labels: array-like of shape (N,) with cluster labels assigned by the model.
    :return: dict with keys 'ARI', 'AMI', 'V-measure'.
    """
    return {
        "ARI": adjusted_rand_score(true_labels, pred_labels),
        "AMI": adjusted_mutual_info_score(true_labels, pred_labels, average_method='arithmetic'),
        "V-measure": v_measure_score(true_labels, pred_labels)
    }
