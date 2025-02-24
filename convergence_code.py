from Standard_Imports import *

lattice2D = CuPyLattice.load("new_a1_a2_test.pkl")
lattice1D = CuPyLattice.load("a2_10_test.pkl")


def shape(data):

    return data.get().reshape((25, 25))


shaped_a1_array = shape(lattice2D.a1)
shaped_a2_array = shape(lattice2D.a2)
shaped_phi_std_array = shape(lattice2D.phi_std)

column = 6

plt.scatter(
    shaped_a2_array[:, column], shaped_phi_std_array[:, column], label="2D Lattice"
)
plt.scatter(lattice1D.a2.get(), lattice1D.phi_std.get(), label="1D Lattice")
plt.legend()
plt.title("Phi StD vs a2 for 1D and 2D Lattices")
plt.xlabel("a2")
plt.ylabel("Phi StD")

plt.text(
    20,
    0.006,
    f"2D: {lattice2D.iteration_count:.2e} iters, a1={shaped_a1_array[0,column]}",
    fontsize=10,
)
plt.text(
    20,
    0.0055,
    f"1D: {lattice1D.iteration_count:.2e} iters, a1={lattice1D.a1[0]}",
    fontsize=10,
)
plt.savefig("trial_convergence_plot.png")
