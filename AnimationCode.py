"""Animating the updates of the fields under the Metropolis Algorithm"""

from Standard_Imports import *
import matplotlib.animation as animation

Ani_Lattice = Lattice(60, 60)
Ani_Lattice.a2 = 75000


"""Mostly copied from ChatGPT"""

# Create a figure and axis
fig, ax = plt.subplots()
ax.set_axis_off()
cax = ax.matshow(Ani_Lattice.phi_array, cmap="viridis")  # Initial display
plt.colorbar(cax)


# Update function for animation
def update_frame(frame_number):
    # Perform 10 updates for each frame
    for _ in range(1000):
        Ani_Lattice.make_update()  # Your function to update the field

    # Update the data for the new frame
    cax.set_data(Ani_Lattice.phi_array)
    return [cax]


# Create the animation
ani = animation.FuncAnimation(fig, update_frame, frames=200, blit=True)

# Save or show the animation
# To save the animation as a GIF or video file
# ani.save('field_animation.mp4', writer='ffmpeg', fps=30)

# Or to display it directly
plt.show()
