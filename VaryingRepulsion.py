from Standard_Imports import *
import matplotlib.animation as animation
import time


# a1 is a measure of the repulsion
grid = Lattice.load("PointSource.pkl")
grid.energy_count = 0

a1 = grid.a1
grid.fluct_phi *= 10
grid.beta *= 10

a1vals = np.linspace(a1 * 20, a1 * 20, 1)


# For animation
a1_iter = 3000000  # Iterations before a is updated
frame_iter = 15000  # Iterations before frame is updated
iterations = a1_iter * len(a1vals)
frames_per_a1 = a1_iter // frame_iter


frame_number = iterations // frame_iter


# Create a figure and axis
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.set_axis_off()
ax1.set_title(f"Concentration")
cax1 = ax1.matshow(grid.phi_array, cmap="plasma", vmin=0, vmax=2)
fig.colorbar(cax1)


(line1,) = ax2.plot([], [], color="blue")


ax2.set_title("Total Energy vs Iterations")
ax2.set_xlabel("Iterations")
ax2.set_ylabel("Energy")
ax2.set_xlim(0, frame_number)


energy_array = []


a1count = 0


# Update function for animation
def update_frame(frame_count):
    global a1count
    global a1vals
    start_time = time.time()

    if frame_count % frames_per_a1 == 0 and frame_count > 0:
        print("Next a1")

        a1count += 1

        print(f"a1count= {a1count}")
        print(f"Before update:{grid.a1}")
        grid.a1 = a1vals[a1count]
        print(f"After update:{grid.a1}")

    for _ in range(frame_iter):
        grid.make_update()

    energy_array.append(grid.energy_count)

    line1.set_data(range(len(energy_array)), energy_array)

    # Update the data for the new frame
    cax1.set_data(grid.phi_array)
    ax1.set_title(f"Concentration, a1={grid.a1:.2e}")

    # Dynamically update the Y-axis of the energy plot based on current energy
    # Adjusting for large energy values
    if len(energy_array) > 1:  # After the first frame
        ax2.set_ylim(
            1.1 * np.min(energy_array), 1.1 * max(np.max(energy_array), 0)
        )  # Dynamically adjust y-axis to fit energy values

    else:
        ax2.set_ylim(0, 1.1 * energy_array[0])  # Initial adjustment for first frame

    end_time = time.time()
    time_diff = end_time - start_time
    print(
        f"Frame {frame_count + 1}/{frame_number} complete, a1={grid.a1}, time={time_diff:.2f}"
    )

    return [cax1, line1]


# Create the animation
ani = animation.FuncAnimation(fig, update_frame, frames=frame_number, blit=True)

# Save or show the animation
# To save the animation as a GIF or video file
plt.tight_layout()
ani.save("Filename.mp4", writer="ffmpeg", fps=10)
