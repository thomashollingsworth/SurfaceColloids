import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

directory = "Large_Investigating_a2_results"


df = pd.read_csv(os.path.join(directory, "std_vals.csv"))


plt.title("Standard Deviation vs a2 Parameter")
plt.plot(df["Unnamed: 0"], df["uniform_initial"], label="Uniform Initial")
plt.plot(df["Unnamed: 0"], df["clustered_initial"], label="Clustered Initial")
plt.legend(loc="best")

plt.show()
