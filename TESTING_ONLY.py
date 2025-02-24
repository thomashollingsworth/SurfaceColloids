from Standard_Imports import *

data = CuPyLattice.load(
    "/home/teh46/PartIII/SurfaceColloids/temp_variation/betaf_2.00e+02_lattice.pkl"
)
print(f"iterations= {data.iteration_count}")
