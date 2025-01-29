"""Experimental script to group clusters of phi values in the lattice."""

from Standard_Imports import *
from scipy.ndimage import label

lattice = Lattice.load("UPDATE2__iters_200000.pkl")
lattice.draw_fields()
lattice.update_all_stats()


def group_clusters(array, cutoff):

    binary_mask = array >= cutoff

    labeled_array, num_features = label(binary_mask, structure=np.ones((3, 3)))

    cluster_sizes = [
        np.sum(array[labeled_array == i]) for i in range(1, num_features + 1)
    ]

    return binary_mask, labeled_array, num_features, cluster_sizes


binary_mask, labeled_array, num_features, cluster_sizes = group_clusters(
    lattice.phi_array, (lattice.initial_phi / 2)
)

figure = plt.figure()
axes_1 = figure.add_subplot(121)

axes_2 = figure.add_subplot(122)


plot1 = axes_1.matshow(binary_mask, cmap="plasma")

plot2 = axes_2.matshow(labeled_array, cmap="plasma")


figure.suptitle("Clustering")

axes_1.set_axis_off()
axes_2.set_axis_off()


plt.tight_layout()
plt.show()

print(
    f"Number of Clusters:{num_features}\nMean Cluster size:{np.mean(cluster_sizes)}\nCluster Deviations:{np.std(cluster_sizes)}"
)
