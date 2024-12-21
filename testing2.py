from Standard_Imports import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
from scipy.stats import normaltest, norm

directory = "Investigating_a2_results"


# Read the CSV file
df = pd.read_csv(os.path.join(directory, "std_vals.csv"))

# Print the contents
print(df)
