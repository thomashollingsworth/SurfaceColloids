from Standard_Imports import *
import temperature_annealing as ta

"""Creating 2D plots of phi_std vs a1 and a2 for different temperatures
- Decide on a final temp.
- Linearly vary temp. over a load of iterations to its final value and then do a final run at that temp.
"""

directory = "/home/teh46/PartIII/SurfaceColloids/temp_variation"

data = CuPyLattice.load("new_a1_a2_test.pkl")
beta_vals = 10 ** np.array([1, 0.5, -0.5, -1, -1.5, -2]) * 20000


def temp_variation(data, beta_f):
    # Currently at room temp. (beta_0=20000)
    beta_0 = 20000
    beta_f = beta_f

    adjust_iters = 0
    final_iters = 140000

    # Performing the iterations:

    # Optional adjustment iterations
    for i in range(adjust_iters):
        data.make_update()
        if i % 100 == 0:
            beta = ta.linear(beta_0, beta_f, i, adjust_iters)
            data.beta = cp.ones_like(data.beta) * beta

    data.beta = cp.ones_like(data.beta) * beta_f

    for i in range(final_iters):
        data.make_update()

    # Creating std plot
    x = data.a1.get().reshape(25, 25)
    y = data.a2.get().reshape(25, 25)
    z = np.log(data.phi_std.get().reshape(25, 25))

    plt.figure()
    heatmap = plt.contourf(x, y, z, cmap="plasma")
    plt.colorbar(heatmap)
    plt.xlabel("a1")
    plt.ylabel("a2")
    kb_scale = 1.68e-7
    plt.title(f"Log(Colloid Deviation), T={(1/(kb_scale*beta_f)):.2e}K")

    # Save final results to a directory

    directory = "temp_variation"
    os.makedirs(directory, exist_ok=True)

    filename = f"betaf_{beta_f:.2e}_stdplot.png"

    plt.savefig(os.path.join(directory, filename))

    # Creating energy/convergence plot

    x1 = data.a1.get().reshape(25, 25)
    y1 = data.a2.get().reshape(25, 25)
    z1 = data.calc_absolute_energy().get().reshape(25, 25)

    plt.figure()
    heatmap1 = plt.contourf(x1, y1, z1, cmap="plasma")
    plt.colorbar(heatmap)
    plt.xlabel("a1")
    plt.ylabel("a2")
    kb_scale = 1.68e-7
    plt.title(f"Energy, T={(1/(kb_scale*beta_f)):.2e}K")

    # Save final results to a directory

    directory = "temp_variation"
    os.makedirs(directory, exist_ok=True)

    filename1 = f"betaf_{beta_f:.2e}_energyplot.png"

    plt.savefig(os.path.join(directory, filename1))

    # Saving lattice
    data.save_lattice(os.path.join(directory, f"betaf_{beta_f:.2e}_lattice.pkl"))


for beta_f in beta_vals:
    temp_variation(data, beta_f)
