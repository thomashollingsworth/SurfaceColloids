"""Runnning Metropolis Algorithm with different initial conditions on multiple CPUs synchronously"""

from Standard_Imports import *
import multiprocessing

num_cpus = multiprocessing.cpu_count()
print(f"Number of CPUs: {num_cpus}")

a2_range = np.linspace(13, 260, num_cpus)
a2_range = np.around(a2_range)  # round to nearest integer


"""Will run one Metropolis Algorithm on each CPU with different a2 conditions
- Will perform one run starting with a uniform background phi
- Will save the final lattice class instance as a pkl file and will record the standard deviation (order parameter) in a dataframe


- A second run will be performed with the final phi array of the highest a2 value as the intial condition for all a2 values
- Will save the final lattice class instance as a pkl file and will record the standard deviation (order parameter) in a dataframe
"""
# Create dataframe to store std values

std_vals = pd.DataFrame


indices = [f"{a2_range[i]}" for i in range(num_cpus)]


std_vals = pd.DataFrame(index=indices)

# Writing the process for each CPU


def run_metropolis_uniform_initial(a2_val):
    # Choose total iterations and logging iterations
    total_iters = 2000000
    logging_iters = 100000  # How often progress is loggeed to output

    # For logging purposes
    process_name = multiprocessing.current_process().name
    process_id = os.getpid()

    # Create lattice
    testgrid = Lattice(25, 25)

    # Set a2 value
    testgrid.a2 = a2_val

    # Set start and end temperatures, update temperature periodically
    beta_0 = 500
    beta_f = 200000  # ROOM TEMP
    beta_update_iters = (
        logging_iters  # Could change this to update temperature more frequently
    )
    start_time = time.time()

    # Running algorithm
    for i in range(total_iters):

        testgrid.make_update()
        if i % logging_iters == 0 and i != 0:
            end_time = time.time()
            testgrid.beta = temp_anneal.power_law(
                beta_0, beta_f, i, total_iters, 4
            )  # Increasing power means more time at high temp!
            # Log progress
            print(
                f"{process_name}({process_id}):Comp.Iter.{i} in {end_time - start_time:.3f}s (uniform, a2={a2_val})"
            )

            start_time = time.time()  # reset timer

    # Save final results to a directory
    directory = "Large_Investigating_a2_results"
    os.makedirs(directory, exist_ok=True)

    filename = f"a2_{testgrid.a2}_uniform.pkl"

    testgrid.save_lattice(os.path.join(directory, filename))

    return testgrid.phi_std


if __name__ == "__main__":
    # Run the Metropolis Algorithm on each CPU
    with multiprocessing.Pool(num_cpus) as pool:
        std_vals["uniform_initial"] = pool.map(run_metropolis_uniform_initial, a2_range)

    # Create a function to run Metropolis Algorithm with the final phi array of the highest a2 value as the initial condition
    def run_metropolis_clustered_initial(a2_val):
        # Locates the final phi array of the highest a2 value for use as initial condition
        highest_a2 = np.max(a2_range)
        directory = "Large_Investigating_a2_results"
        file_location = os.path.join(directory, f"a2_{highest_a2}_uniform.pkl")

        # Choose total iterations and logging iterations
        total_iters = 2000000
        logging_iters = 100000  # How often progress is loggeed to output

        # For logging purposes
        process_name = multiprocessing.current_process().name
        process_id = os.getpid()

        # Create lattice
        testgrid = Lattice.load(file_location)
        # Reset energy count, energy array and std_array
        testgrid.energy_count = 0
        testgrid.energy_array = []
        testgrid.phi_std_array = []

        # Set a2 value
        testgrid.a2 = a2_val

        # Set start and end temperatures, update temperature periodically
        beta_0 = 1000
        beta_f = 200000  # ROOM TEMP
        beta_update_iters = (
            logging_iters  # Could change this to update temperature more frequently
        )

        start_time = time.time()
        # Running algorithm
        for i in range(total_iters):

            testgrid.make_update()
            if i % logging_iters == 0 and i != 0:
                end_time = time.time()
                testgrid.beta = temp_anneal.power_law(
                    beta_0, beta_f, i, total_iters, 4
                )  # Again, could change this to update temperature more frequently
                # Log progress
                print(
                    f"{process_name}({process_id}):Comp.Iter.{i} in {end_time - start_time:.3f}s (clustered, a2={a2_val})"
                )

                start_time = time.time()  # reset timer

        filename = f"a2_{testgrid.a2}_clustered.pkl"

        testgrid.save_lattice(os.path.join(directory, filename))

        return testgrid.phi_std

    with multiprocessing.Pool(num_cpus) as pool:
        std_vals["clustered_initial"] = pool.map(
            run_metropolis_clustered_initial, a2_range
        )

    # Save the results to a csv file
    directory = "Large_Investigating_a2_results"
    os.makedirs(directory, exist_ok=True)
    std_vals.to_csv(os.path.join(directory, "std_vals.csv"))
