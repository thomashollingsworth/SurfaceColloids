"""Performing Various Analyses on the Lattice Data"""

from Standard_Imports import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
from scipy.stats import normaltest, norm


directory = "Large_Investigating_a2_results"
data = "testgrid.pkl"

new_lattice = Lattice.load(data)

phi_array = new_lattice.phi_array
phi_std_vals = new_lattice.phi_std_array

# Perform the Fourier transform
fft_result = fftshift(fft2(phi_array))
magnitude_spectrum = np.abs(fft_result)

# Set the zero frequency component to zero
magnitude_spectrum[magnitude_spectrum == np.max(magnitude_spectrum)] = 0

# Plot the original concentration data
plt.figure()
plt.imshow(phi_array, cmap="plasma")
plt.colorbar(label="Concentration")
plt.title("Concentration of Particles")
plt.show()

# Histogram Analysis
mean_phi = np.mean(phi_array)
std_phi = np.std(phi_array)
plt.figure()
plt.hist(phi_array.flatten(), bins=50, density=True, alpha=0.6, color="g")

# Plot the normal distribution fit
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mean_phi, std_phi)
plt.plot(x, p, "k", linewidth=2)
plt.title("Histogram of Concentration Values with Normal Distribution Fit")
plt.xlabel("Concentration")
plt.ylabel("Density")
plt.show()

# Statistical Test (D'Agostino and Pearson's test)
stat, p_value = normaltest(phi_array.flatten())
print(f"D'Agostino and Pearson's Test Statistic: {stat}, P-value: {p_value}")


# Fourier Transform Analysis
plt.figure()
plt.imshow(np.log1p(magnitude_spectrum), cmap="plasma")
plt.colorbar(label="Log Magnitude")
plt.title("Fourier Transform of Concentration")
plt.xlabel("Frequency X")
plt.ylabel("Frequency Y")
plt.show()

# Residual Analysis
residuals = phi_array - mean_phi
plt.figure()
plt.imshow(residuals, cmap="plasma")
plt.colorbar(label="Residuals")
plt.title("Residuals (Concentration - Mean Concentration)")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# Plot the standard deviation of concentration vs iterations
plt.plot(np.arange(len(phi_std_vals)), phi_std_vals)
plt.xlabel("Iterations")
plt.ylabel("Standard Deviation of Concentration")
plt.title("Standard Deviation of Concentration vs Iterations")
plt.tight_layout()
plt.show()

# Plot energy vs iterations
plt.figure()
plt.plot(
    np.arange(len(new_lattice.energy_array))[1000000:],
    new_lattice.energy_array[1000000:],
)

plt.title("Energy vs Iterations")
plt.xlabel("Iterations")
plt.ylabel("Energy")
plt.show()
