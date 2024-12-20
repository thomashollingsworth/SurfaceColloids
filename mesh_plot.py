import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from matplotlib import cm


def mesh_plot(lattice_instance, z_stretch=50000):
    """Creates a 3D mesh plot of the height of the domain overlayed with a colour map corresponding to the concentration.

    Args:
        lattice_instance (_type_): The instance of the lattice class you want to plot the fields for.
    """

    dx = 70
    dy = 70
    height_array = lattice_instance.h_array

    norm = Normalize(
        vmin=np.min(lattice_instance.phi_array), vmax=np.max(lattice_instance.phi_array)
    )
    colors = cm.plasma(norm(lattice_instance.phi_array))

    x = np.arange(0, lattice_instance.num_columns * dx, dx)
    y = np.arange(0, lattice_instance.num_rows * dy, dy)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    surface = ax.plot_surface(X, Y, height_array, facecolors=colors, edgecolor="k")

    # Set equal scaling for all axes
    max_range = (
        np.array(
            [
                X.max() - X.min(),
                Y.max() - Y.min(),
                height_array.max() - height_array.min(),
            ]
        ).max()
        / 2.0
    )
    mid_x = (X.max() + X.min()) * 0.5
    mid_y = (Y.max() + Y.min()) * 0.5
    mid_z = (height_array.max() + height_array.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range * 1 / z_stretch, mid_z + max_range * 1 / z_stretch)

    ax.set_axis_off()

    plt.show()
