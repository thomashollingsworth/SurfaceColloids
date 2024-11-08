"""Animating the updates of the fields under the Metropolis Algorithm"""

from Standard_Imports import *
import matplotlib.animation as animation

testgrid1 = Lattice(20, 20)


# Iniital phi conditions
num_lattice_points = testgrid1.num_columns * testgrid1.num_rows


new_initial_phi_array = np.zeros((testgrid1.num_rows, testgrid1.num_columns))
new_initial_phi_array[testgrid1.num_rows // 2, testgrid1.num_columns // 2] = (
    num_lattice_points * testgrid1.initial_phi
)

# Set initial conditions
# testgrid1.phi_array = new_initial_phi_array


testgrid1.beta = 0.1
testgrid1.a2 *= 50
# testgrid1.fluct_phi *= 20


# For animation
interval = 5000
iterations = 250000

frame_number = iterations // interval


"""Mostly copied from ChatGPT"""

# Create a figure and axis
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.set_axis_off()
ax1.set_title(f"Concentration, Beta={testgrid1.beta}")
cax1 = ax1.matshow(testgrid1.phi_array, cmap="plasma")  # Initial display
fig.colorbar(cax1)

(line1,) = ax2.plot([], [], color="blue", label=f"Beta={testgrid1.beta}")

ax2.legend()

ax2.set_title("Total Energy vs Iterations")
ax2.set_xlabel("Iterations")
ax2.set_ylabel("Energy")
ax2.set_xlim(0, frame_number)


energy1_array = []


# Update function for animation
def update_frame(frame_count):
    print(f"Frame {frame_count + 1} completed")

    for _ in range(interval):
        testgrid1.make_update()

    energy1_array.append(testgrid1.energy_count)

    line1.set_data(range(len(energy1_array)), energy1_array)

    # Update the data for the new frame
    cax1.set_data(testgrid1.phi_array)

    # Dynamically update the Y-axis of the energy plot based on current energy
    # Adjusting for large energy values
    if len(energy1_array) > 1:  # After the first frame
        ax2.set_ylim(
            -1.1 * np.max(np.abs(energy1_array), 0)
        )  # Dynamically adjust y-axis to fit energy values

    else:
        ax2.set_ylim(0, 1.1 * energy1_array[0])  # Initial adjustment for first frame

    return [cax1, line1]


# Create the animation
ani = animation.FuncAnimation(fig, update_frame, frames=frame_number, blit=True)

# Save or show the animation
# To save the animation as a GIF or video file
plt.tight_layout()
ani.save("TrialAnimation.mp4", writer="ffmpeg", fps=10)

# Or to display it directly
# plt.show()
