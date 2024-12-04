"""Animating the updates of the fields under the Metropolis Algorithm"""

from Standard_Imports import *
import matplotlib.animation as animation

testgrid1 = Lattice(50, 50)


# Creating a 'point source' initial condition
num_lattice_points = testgrid1.num_columns * testgrid1.num_rows


new_initial_phi_array = np.zeros((testgrid1.num_rows, testgrid1.num_columns))
new_initial_phi_array[testgrid1.num_rows // 2, testgrid1.num_columns // 2] = (
    num_lattice_points * testgrid1.initial_phi
)

# Set point source initial conditions

testgrid1.phi_array = new_initial_phi_array

testgrid1.fluct_h *= 20
testgrid1.beta = 0.005

testgrid1.a2 = 10000 * 5


# For animation
interval = 1000
iterations = 50000

frame_number = iterations // interval


# Create a figure and axis
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.set_axis_off()
ax1.set_title(f"Concentration")
cax1 = ax1.matshow(testgrid1.phi_array, cmap="plasma", vmin=0, vmax=2)
fig.colorbar(cax1)


(line1,) = ax2.plot([], [], color="blue")


ax2.set_title("Total Energy vs Iterations")
ax2.set_xlabel("Iterations")
ax2.set_ylabel("Energy")
ax2.set_xlim(0, frame_number)


energy_array = []


# Update function for animation
def update_frame(frame_count):
    print(f"Frame {frame_count + 1}/{iterations//interval} complete")

    for _ in range(interval):
        testgrid1.make_update()

        # Linearly decreasing size of fluctuations
        # testgrid1.fluct_phi = max(testgrid1.fluct_phi - ((10 - 0.01) / interval), 0.01)
        # testgrid1.fluct_h = max(testgrid1.fluct_h - ((10 - 0.01) / interval), 0.01)

    energy_array.append(testgrid1.energy_count)

    line1.set_data(range(len(energy_array)), energy_array)

    # Update the data for the new frame
    cax1.set_data(testgrid1.phi_array)

    # Dynamically update the Y-axis of the energy plot based on current energy
    # Adjusting for large energy values

    if len(energy_array) == 0:

        ax2.set_ylim(0, 1.1 * energy_array[0])  # Initial adjustment for first frame

    else:  # After the first frame
        ax2.set_ylim(
            energy_array[frame_count]
            - 5 * np.abs(energy_array[frame_count] - energy_array[frame_count - 1]),
            energy_array[frame_count]
            + 5 * np.abs(energy_array[frame_count] - energy_array[frame_count - 1]),
        )  # Dynamically adjust y-axis to fit energy values

    return [cax1, line1]


# Create the animation
ani = animation.FuncAnimation(fig, update_frame, frames=frame_number, blit=True)

# Save or show the animation
# To save the animation as a GIF or video file
plt.tight_layout()

ani.save("TESTINGANI.mp4", writer="ffmpeg", fps=10)


# testgrid1.save_lattice("Filename.pkl")
