from Standard_Imports import *

full_testgrid = Lattice.load("testgrid.pkl")
full_h_array = full_testgrid.h_array
full_phi_array = full_testgrid.phi_array

n = 20

new_h_array = np.zeros((4 * n + 2, 4 * n + 2))
new_phi_array = np.zeros((4 * n + 2, 4 * n + 2))

new_h_array = full_h_array[1 : 4 * n + 3, 1 : 4 * n + 3]
new_phi_array = full_phi_array[1 : 4 * n + 3, 1 : 4 * n + 3]


new_h_array = np.pad(new_h_array, pad_width=1, mode="constant", constant_values=0)
new_phi_array = np.pad(
    new_phi_array,
    pad_width=1,
    mode="constant",
    constant_values=full_testgrid.initial_phi,
)

small_testgrid = Lattice(n, n)
small_testgrid.h_array = new_h_array
small_testgrid.phi_array = new_phi_array
small_testgrid.set_initial_conditions()
small_testgrid.draw_fields()
small_testgrid.save_lattice("small_testgrid.pkl")
