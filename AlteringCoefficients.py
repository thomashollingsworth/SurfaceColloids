"""Testing how h and phi fields are affected by altering coefficients"""

from Standard_Imports import *

testgrid = Lattice(5, 5)


num_lattice_points = testgrid.num_columns * testgrid.num_rows


new_initial_phi_array = np.zeros((testgrid.num_rows, testgrid.num_columns))
new_initial_phi_array[testgrid.num_rows // 2, testgrid.num_columns // 2] = (
    num_lattice_points * testgrid.initial_phi
)
testgrid.phi_array = new_initial_phi_array
testgrid.a2 *= 3
testgrid.fluct_phi *= 10

interval = 1000
iterations = 250000


energy_array = np.zeros(iterations)[::interval]
phi_std_array = np.zeros(iterations)[::interval]

for i in range(iterations):

    # Could edit any of the parameters during the iteration process
    # e.g. could dynamically alter beta using a Simulated Annealing algorithm
    # testgrid.beta = 0.0001 + 0.2069 * np.log(1 + i)

    testgrid.beta = 0.03

    testgrid.make_update()  # repetitively updating the lattice

    if (
        i % interval == 0
    ):  # periodically recording the energy changes/any other quantities of interest
        testgrid.update_stats()  # have to explicitly call update_stats() isn't included in make_update()
        energy_array[i // interval] = testgrid.energy_count
        phi_std_array[i // interval] = testgrid.phi_std
        print(f"Iteration {i} Completed")

# Plotting results

testgrid.draw_fields(save=False)
# print(f"Mean h:{newgrid.h_mean}\nMean phi:{newgrid.phi_mean}\nStd. Phi: {newgrid.phi_std}")

fig, ax1 = plt.subplots()

color1 = "tab:blue"
ax1.set_xlabel("Iterations")
ax1.set_ylabel("Energy", color=color1)
ax1.plot(np.arange(iterations)[::interval], energy_array, color=color1)
ax1.tick_params(axis="y", labelcolor=color1)

"""
ax2 = ax1.twinx()

color2 = "tab:green"
ax2.set_ylabel("Deviation of phi", color=color2)
ax2.plot(np.arange(iterations)[::interval], phi_std_array, color=color2)
ax2.tick_params(axis="y", labelcolor=color2)
"""

fig.suptitle("Total Energy")
plt.show()

# testgrid.save_lattice("Lattice_Name.pkl")
