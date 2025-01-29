from Standard_Imports import *

testgrid = Lattice(100, 100)

adjustment_iters = 200000
finalval_iters = 50000  # 200000
logging_iters = 1000  # How often progress is logged to output


beta_0 = 5000
beta_f = 200000  # ROOM TEMP

a2_0 = 1500
a2_f = 250
n = 1.5

start_time = time.time()
testgrid.a2 = a2_0
testgrid.set_initial_conditions()

for i in range(adjustment_iters):

    testgrid.make_update()
    if i % logging_iters == 0 and i != 0:
        end_time = time.time()
        testgrid.beta = temp_anneal.power_law(beta_0, beta_f, i, adjustment_iters, 4)

        # Log progress
        print(f"Comp.Iter.{i} in {end_time - start_time:.3f}s")

        start_time = time.time()  # reset timer
        if (i / logging_iters) % 10 == 0 and i != 0:
            testgrid.a2 = a2_0 + (a2_f - a2_0) * (i / adjustment_iters) ** n
            testgrid.energy_count = testgrid.calc_absolute_energy()

start_time = time.time()
testgrid.a2 = a2_f
testgrid.beta = beta_f
testgrid.energy_count = testgrid.calc_absolute_energy()

for i in range(finalval_iters):
    testgrid.make_update()
    if i % logging_iters == 0 and i != 0:
        end_time = time.time()
        testgrid.energy_count = testgrid.calc_absolute_energy()
        # Log progress
        print(f"Comp.Iter.{i} in {end_time - start_time:.3f}s")

        start_time = time.time()  # reset timer

testgrid.save_lattice("testgrid2.pkl")
fig, axs = plt.subplots(2, 1)
axs[0].plot(np.arange(len(testgrid.energy_array)), testgrid.energy_array)
axs[0].set_title("Energy vs Iteration")
axs[1].plot(np.arange(len(testgrid.phi_std_array)), testgrid.phi_std_array)
axs[1].set_title("Phi Std Dev vs Iteration")
plt.tight_layout()
plt.show()
testgrid.draw_fields()
print(testgrid.calc_absolute_energy())
