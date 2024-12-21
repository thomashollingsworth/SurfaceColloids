import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

directory = "Investigating_a2_results"


df = pd.read_csv(os.path.join(directory, "std_vals.csv"))
print(df.keys())
plt.plot(df["Unnamed: 0"], df["uniform_initial"], label="Uniform Initial")
plt.plot(df["Unnamed: 0"], df["clustered_initial"], label="Clustered Initial")

plt.show()
