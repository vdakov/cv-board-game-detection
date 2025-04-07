import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def cluster_masks(anns, min_samples=3, eps=0.5):
    features = np.array([ann["area"] for ann in anns]).reshape(-1, 1)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(features_scaled)

    for i, ann in enumerate(anns):
        ann["cluster"] = clustering.labels_[i]

    return anns
