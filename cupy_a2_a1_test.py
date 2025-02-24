from Standard_Imports import *

print(f"Num. GPUs:{cp.cuda.runtime.getDeviceCount()}")

num_trials = 625
testlattice = CuPyLattice(25, num_trials)  # Dimension=..., Num_trials=...

# Determining Initial Conditions

testlattice.a2 = cp.repeat(cp.linspace(0, 50, 25), 25)
testlattice.a1 = cp.tile(cp.linspace(0, 5, 25), 25)


# Creating Point Source for Phi arrays
total_phi = testlattice.initial_phi * 16 * (testlattice.dimension + 1) ** 2

pointsource = cp.zeros_like(testlattice.phi_array)
pointsource[:, 2 * (testlattice.dimension + 1), 2 * (testlattice.dimension + 1)] = (
    total_phi
)
testlattice.phi_array = pointsource


total_iters = 2500000
logging_iters = 10000

# TEMPERATURE ANNEALING
beta_0 = 200000 * 0.01
beta_f = 200000  # ROOM TEMP = 200000


start_time = time.time()

for i in range(total_iters):
    testlattice.make_update()
    if i % logging_iters == 0 and i != 0:
        end_time = time.time()
        print(f"Completed {logging_iters} iterations in {(end_time-start_time):.3g}s")
        start_time = end_time


testlattice.save_lattice("smalla2_a1_a2_test.pkl")
