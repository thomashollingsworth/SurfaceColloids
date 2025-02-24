from Standard_Imports import *

data = CuPyLattice.load("5xphi_a2_test.pkl")


def a2_plots():
    """a2 energy and phi plots"""
    energy = data.calc_absolute_energy().get()
    a2 = data.a2.get()
    std = data.phi_std.get()

    fig, axs = plt.subplots(2, 1)
    axs[0].plot(a2, energy, color="blue", linewidth=0.5)

    axs[0].set_xlabel("a2")
    axs[0].set_ylabel("Eqm. Energy")

    axs[1].plot(
        a2,
        std,
        color="blue",
        linewidth=0.5,
    )
    axs[1].set_xlabel("a2")
    axs[1].set_ylabel("Eqm. Standard Deviation")
    fig.suptitle("Variation with Colloid Density (a2)")
    fig.tight_layout()
    plt.savefig("5xphi_a2_plot.png")


def phi_plots():
    """Phi heatmap plots"""
    directory = "5xphi_heatmaps"
    os.makedirs(directory, exist_ok=True)

    for i in range(data.num_trials):
        filename = f"heatmap_a2_{data.a2[i]:.2f}.png"

        filepath = os.path.join(directory, filename)

        data.draw_fields(i, True, filepath)


phi_plots()
a2_plots()
