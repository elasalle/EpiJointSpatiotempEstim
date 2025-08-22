import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from collections import defaultdict

import geopandas as gpd


def spectral_clustering_from_laplacian(L_est, max_clusters=10, random_state=42):
    """
    Perform spectral clustering from a Laplacian matrix.


    Parameters
    ----------
    L_est : np.ndarray
    Laplacian matrix (n x n).
    max_clusters : int
    Maximum number of clusters to check for eigengap heuristic.
    random_state : int
    Random state for reproducibility.

    Returns
    -------
    labels : np.ndarray
    Cluster labels for each node.
    k : int
    Estimated number of clusters.
    """
    eigvals, eigvecs = np.linalg.eigh(L_est)
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Eigengap heuristic
    gaps = np.diff(eigvals[:max_clusters+1])
    k = np.argmax(gaps) + 1
    print("{} clusters will be computed".format(k))

    X = eigvecs[:, :k]
    X = normalize(X)
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=random_state)
    labels = kmeans.fit_predict(X)

    return labels, k

def print_clusters(labels, countries):
    """
    Print clusters of countries given labels.
    """
    clusters = defaultdict(list)
    for country, label in zip(countries, labels):
        clusters[label].append(country)

    for cluster_id, members in sorted(clusters.items()):
        print(f"Cluster {cluster_id}:")
        print(members)


# Map the clusters

plt.rcParams['hatch.linewidth'] = 0.5  # default is 1.0

def plot_country_clusters(countries, cluster_labels, cmap_name="tab10"):
    # Load world map
    url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    world = gpd.read_file(url)
    rename_map = {"POP_EST": "pop_est", "CONTINENT": "continent", "ADMIN": "name", "ADM0_A3": "iso_a3"}
    world = world.rename(columns={k: v for k, v in rename_map.items() if k in world.columns})

    # Map country -> cluster
    country_to_label = dict(zip(countries, cluster_labels))

    # Subset to selected countries only
    subset = world[world["name"].isin(countries)].copy()
    subset["cluster"] = subset["name"].map(country_to_label)

    # Separate isolated and non-isolated
    isolated = subset[subset["cluster"] == 0]
    non_isolated = subset[subset["cluster"] != 0]

    # Reproject for Europe
    subset = subset.to_crs("EPSG:3035")
    isolated = isolated.to_crs("EPSG:3035")
    non_isolated = non_isolated.to_crs("EPSG:3035")

    # Use discrete colormap indexing
    cmap = plt.get_cmap(cmap_name)
    unique_clusters = sorted(non_isolated["cluster"].unique())
    color_map = {cl: cmap(cl % cmap.N) for cl in unique_clusters}

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(5, 5))

    # Non-isolated clusters in tab10 colors directly
    if not non_isolated.empty:
        for cl, group in non_isolated.groupby("cluster"):
            group.plot(
                ax=ax,
                color=color_map[cl],
                edgecolor="black",
                label=f"Cluster {cl}",
                linewidth=0.5
            )

    # Isolated clusters in dashed hatch, no color
    if not isolated.empty:
        isolated.plot(
            ax=ax,
            facecolor="white",
            edgecolor="black",
            hatch="/////",
            label="Isolated",
            linewidth=0.5
        )

    # Discrete legend instead of continuous colorbar
    if not isolated.empty:
        handles = [Patch(facecolor="white", edgecolor="black",
                            hatch="/////", label="Isolated", linewidth=0.5)]

    handles += [Patch(facecolor=color_map[cl], edgecolor="black", linewidth=0.5,
                     label=f"Cluster {cl}")
               for cl in unique_clusters]



    ax.legend(handles=handles, loc='center left', bbox_to_anchor=[1,0.5], labelspacing=0.2, handlelength=1.5, handletextpad=0.4)


    # --- Zoom to subset---
    # minx, miny, maxx, maxy = subset.total_bounds
    minx, maxx = 1.4e6, 7.3e6
    miny, maxy = -0.4e6, 5.5e6
    print(minx, miny, maxx, maxy)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    # ax.set_title("Clustered Countries", fontsize=14)
    ax.set_axis_off()

    plt.tight_layout()
    return fig