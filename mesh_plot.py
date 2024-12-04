import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from matplotlib import cm


def mesh_plot(lattice_instance):
    """Creates a 3D mesh plot of the height of the domain overlayed with a colour map corresponding to the concentration.

    Args:
        lattice_instance (_type_): The instance of the lattice class you want to plot the fields for.
    """

    height_array = lattice_instance.h_array

    norm = Normalize(
        vmin=np.min(lattice_instance.phi_array), vmax=np.max(lattice_instance.phi_array)
    )
    colors = cm.plasma(norm(lattice_instance.phi_array))

    x = np.arange(lattice_instance.num_columns)
    y = np.arange(lattice_instance.num_rows)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    surface = ax.plot_surface(X, Y, height_array, facecolors=colors, edgecolor="k")

    ax.set_axis_off()

    plt.show()
