from Standard_Imports import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
from scipy.stats import normaltest, norm

new_lattice = Lattice.load("B__iters_800000.pkl")
print(f"a3:{new_lattice.a3}, a2:{new_lattice.a2}, a1:{new_lattice.a1}")
print(f"Dimensions:{new_lattice.num_rows}x{new_lattice.num_columns}")

"""
a1=12,  # coefficient of electrostatic interactions (repulsion)
a2=13,  # coefficient of extra density of colloids
a3=2000,  # coeff of surface tension
a4=1.3,  # coefficient of density difference of liquids??
        """
