"""Animating the updates of the fields under the Metropolis Algorithm"""

from Standard_Imports import *
import matplotlib.animation as animation


lattice_name = "New"  # , "Uniform","PointSource" etc.

# ________________________________________________________
# testgrid = Lattice.load("a2_260_clustered.pkl")
testgrid = Lattice(25, 25)
testgrid.a2 = 250

testgrid.set_initial_conditions()
# ________________________________________________________
"""
# Creating a 'point source' initial condition
num_lattice_points = testgrid.num_columns * testgrid.num_rows


new_initial_phi_array = np.zeros((testgrid.num_rows, testgrid.num_columns))
new_initial_phi_array[testgrid.num_rows // 2, testgrid.num_columns // 2] = (
    num_lattice_points * testgrid.initial_phi
)

# Set point source initial conditions

testgrid.phi_array = new_initial_phi_array
"""
# ________________________________________________________


# Setting start and end temps
beta_0 = 1000
# beta_f = 5000
beta_f = 200000  # ROOM TEMP


# a2 Values

# testgrid.a1 *= 100000
# testgrid.a3


# For animation
frame_iterations = 1000
total_iterations = 200000

frame_number = total_iterations // frame_iterations


# Create a figure and axis
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.set_axis_off()
ax1.set_title(f"Concentration")
cax1 = ax1.matshow(testgrid.phi_array, cmap="plasma", vmin=0, vmax=0.02)
fig.colorbar(cax1)


(line1,) = ax2.plot([], [], color="blue")


ax2.set_title("Total Energy vs Iterations")
ax2.set_xlabel("Iterations")
ax2.set_ylabel("Energy")
ax2.set_xlim(0, frame_number)
plt.title(f"a2={testgrid.a2}")


energy_array = []


# Update function for animation
def update_frame(frame_count, frame_iterations, total_iterations, beta_0, beta_f):

    frame_number = total_iterations // frame_iterations
    start_time = time.time()

    testgrid.beta = temp_anneal.logarithmic(beta_0, beta_f, frame_number, frame_count)

    for _ in range(frame_iterations):

        testgrid.make_update()

    energy_array.append(testgrid.energy_count)

    line1.set_data(range(len(energy_array)), energy_array)

    # Update the data for the new frame
    cax1.set_data(testgrid.phi_array)

    margin = 0.1 * (np.max(energy_array) - np.min(energy_array))
    ax2.set_ylim(np.min(energy_array) - margin, np.max(energy_array) + margin)

    end_time = time.time()

    print(
        f"Frame {frame_count + 1}/{frame_number} completed in {end_time - start_time:.3f} seconds"
    )
    return [cax1, line1]


# Create the animation
ani = animation.FuncAnimation(
    fig,
    update_frame,
    frames=frame_number,
    fargs=(frame_iterations, total_iterations, beta_0, beta_f),
    blit=True,
)

# Save or show the animation
# To save the animation as a GIF or video file
plt.tight_layout()
ani.save(
    f"{lattice_name}_iters_{total_iterations}.mp4",
    writer="ffmpeg",
    fps=10,
)


testgrid.save_lattice(f"{lattice_name}_iters_{total_iterations}.pkl")
