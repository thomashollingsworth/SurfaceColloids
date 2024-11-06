"""This file is exclusively used to store the Lattice Class
Any other code i.e. testing debugging must be kept under the, if __name__=="__main__":"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.random as random

# Used for saving data
import csv
import datetime
import pickle


class Lattice:

    def __init__(
        self,
        n=5,  # number of rows of 4x4 grids
        m=5,  # number of columns of 4x4 grids
        initial_phi=0.1,  # Initial magnitude of uniform phi distribution
        fluct_h=0.001,  # Scaling parameter for h fluctuations
        fluct_phi=0.001 * 10,  # Scaling parameter for phi fluctuations
        a1=1000 * 1.5,  # coefficient of electrostatic interactions (repulsion)
        a2=10000 * 3,  # coefficient of extra density of colloids
        a3=1000000,  # coeff of surface tension
        a4=10000,  # coefficient of density difference of liquids??
        beta=1,  # 1/T term
    ) -> None:

        # Counters for the number of make_update() iterations and total energy change of system
        self.energy_count = 0
        self.iteration_count = 0

        # Initialising coefficients/parameters as modifiable class attributes
        self.n = n
        self.m = m
        self.num_rows = 4 * (self.n + 1)
        self.num_columns = 4 * (self.m + 1)

        self.initial_phi = initial_phi
        self.fluct_h = fluct_h
        self.fluct_phi = fluct_phi

        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4

        self.beta = beta

        emptyarray = np.zeros(
            (4 * (self.n + 1), 4 * (self.m + 1))
        )  # Just here for convenience
        onesarray = np.ones((4 * (self.n + 1), 4 * (self.m + 1)))

        # h and phi arrays are useful attributes that can be accessed outside of the class
        # h and phi initially set to uniform 0 and initial_phi respectively
        self.h_array = emptyarray
        self.phi_array = self.initial_phi * onesarray

        # Array attributes for storing the indices and values that will be used in each update
        # These aren't likely to be useful outside of the class
        # They have been set as private attributes using __ notation

        self.__start_rowindex = np.zeros(self.m * self.n)
        self.__start_columnindex = np.zeros(self.m * self.n)
        self.__h_rowindex = np.zeros(self.m * self.n)
        self.__h_columnindex = np.zeros(self.m * self.n)
        self.__phi_rowindex = np.zeros(self.m * self.n)
        self.__phi_columnindex = np.zeros(self.m * self.n)

        self.__new_h = np.zeros(3 * self.m * self.n)
        self.__new_phi = np.zeros(3 * self.m * self.n)
        self.__energy_change = np.zeros(self.m * self.n)

        self.__all_rowindex = np.concatenate(
            (self.__start_rowindex, self.__h_rowindex, self.__phi_rowindex)
        )
        self.__all_columnindex = np.concatenate(
            (self.__start_columnindex, self.__h_columnindex, self.__phi_columnindex)
        )

        # Initialises attributes that track mean and std of h and phi arrays
        self.h_mean, self.h_std = np.mean(self.h_array), np.std(self.h_array)
        self.phi_mean, self.phi_std = np.mean(self.phi_array), np.std(self.phi_array)

        # Arrays that can store the move at each turn (redundant attribute)
        self.__start_array = emptyarray
        self.__move_array = emptyarray

    def draw_fields(self, save=False):
        """Displays the current h_array and phi_array using plt.matshow.

        Args:
            save (bool, optional):Option to save a png of the arrays and a csv file containing the coefficients used. Defaults to False.
        """
        # draws the current h and phi arrays

        figure = plt.figure()
        axes_h = figure.add_subplot(121)
        axes_phi = figure.add_subplot(122)

        h_plot = axes_h.matshow(self.h_array, cmap="plasma")
        phi_plot = axes_phi.matshow(self.phi_array, cmap="plasma")

        axes_h.set_title("Height")
        axes_phi.set_title("Concentration")
        axes_h.set_axis_off()
        axes_phi.set_axis_off()

        figure.colorbar(h_plot)
        figure.colorbar(phi_plot)

        plt.tight_layout()

        if save:
            self.update_stats()

            timestamp = datetime.datetime.now().strftime(
                "%d_%H%M"
            )  # Used to name/identify the saved files by their time of creation

            plt.savefig(f"{timestamp}_arrays.png", dpi=300)  # Saving a png of arrays

            # Save data of choice to a CSV file

            data = f"N,M={self.n},{self.m}\nIterations={self.iteration_count}\n\nCoefficients:\na1={self.a1}\na2={self.a2} \na3={self.a3}\na4={self.a4}\ninitial_phi={self.initial_phi}\nfluctuation phi,h={self.fluct_phi},{self.fluct_h}"

            with open(f"{timestamp}_data.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([data])

        plt.show()

    def update_stats(self):
        """Updates the height and phi mean and standard deviation attributes.
        [To save computation time, this method is not included in make_update()]"""

        self.h_mean, self.h_std = np.mean(self.h_array), np.std(self.h_array)
        self.phi_mean, self.phi_std = np.mean(self.phi_array), np.std(self.phi_array)

    def _update_move_indices(self):
        """Chooses which lattice points to try and alter, this is called within make_update():

        - Randomly chooses a starting lattice point to update from within a 4x4 square
         - Randomly chooses one of this point's 8 nearest neighbours for the h-field to 'move' to and one for the phi-field to 'move' to
          - Gathers the relevant indices for these lattice points and updates the various index attributes accordingly
        """

        start_choice = np.random.randint(
            0, 16
        )  # Choosing starting lattice point in a 4x4 square

        move_choice = [0, 1, 2, 3, 5, 6, 7, 8]  # Used to determine nearest neighbour

        # Index manipulation
        h_moves = np.random.choice(move_choice, self.n * self.m)
        phi_moves = np.random.choice(move_choice, self.n * self.m)

        h_rowmoves, h_columnmoves = (np.floor(h_moves / 3) - 1).astype(int), (
            (h_moves % 3) - 1
        ).astype(int)

        phi_rowmoves, phi_columnmoves = (np.floor(phi_moves / 3) - 1).astype(int), (
            (phi_moves % 3) - 1
        ).astype(int)

        rowshift, columnshift = int(np.floor(start_choice / 4)), (start_choice % 4)

        rowindex = np.arange((2 + rowshift), (2 + 4 * self.n), 4)
        columnindex = np.arange((2 + columnshift), (2 + 4 * self.m), 4)

        startrowindex, startcolumnindex = np.meshgrid(rowindex, columnindex)

        start_rowindex, start_columnindex = startrowindex.flatten(
            order="C"
        ), startcolumnindex.flatten(order="C")

        phi_rowindex = start_rowindex + phi_rowmoves
        phi_columnindex = start_columnindex + phi_columnmoves

        h_rowindex = start_rowindex + h_rowmoves
        h_columnindex = start_columnindex + h_columnmoves

        # Update all the index attributes

        self.__start_rowindex = start_rowindex  # Starting lattice points
        self.__start_columnindex = start_columnindex

        self.__h_rowindex = h_rowindex  # Lattice points that h will 'move' to
        self.__h_columnindex = h_columnindex

        self.__phi_rowindex = phi_rowindex  # Lattice points that phi will 'move' to
        self.__phi_columnindex = phi_columnindex

        self.__all_rowindex = np.concatenate(
            (
                self.__start_rowindex,
                self.__h_rowindex,
                self.__phi_rowindex,
            )  # Collecting all lattice points that will need to be updated
        )
        self.__all_columnindex = np.concatenate(
            (self.__start_columnindex, self.__h_columnindex, self.__phi_columnindex)
        )

    """
    def draw_move(self):
        '''Redundant method that was used in bug fixing. Displays an array indicating all the lattice points that will be updated.'''

        figure = plt.figure()
        axes = figure.add_subplot(111)
        visual = np.zeros((4 * (self.n + 1), 4 * (self.m + 1)))

        visual[self.__start_rowindex, self.__start_columnindex] = 1
        visual[self.__h_rowindex, self.__h_columnindex] = 2
        visual[self.__phi_rowindex, self.__phi_columnindex] = 3

        move_plot = axes.matshow(visual)

        axes.set_title("Random Move (start=1,h=2,phi=3)")

        axes.set_axis_off()

        figure.colorbar(move_plot)

        plt.tight_layout()
        plt.show()
    """

    def _update_h(self):
        """Updates the self.__new_h attribute, called within make_update():

        - A copy of the self.h_array is made
        - A series of random height fluctuations are generated, scaled by self.fluct_h
        - These fluctuations are added to the starting lattice points
        - These fluctuations are subtracted from the corresponding nearest neighbour
        - The new h values in this copied array are saved under the attribute self.__new_h
        """

        # Making a copy of original h-array points that can be manipulated freely
        new_h = np.concatenate(
            (
                self.h_array[self.__start_rowindex, self.__start_columnindex],
                self.h_array[self.__h_rowindex, self.__h_columnindex],
            )
        ).copy()

        fluct_h = self.fluct_h * (
            random.random(np.size(self.__start_rowindex)) - 0.5
        )  # Generating random h fluctuations

        fluct_h = np.concatenate((fluct_h, -fluct_h))

        new_h += fluct_h

        new_h = np.concatenate(
            (new_h, self.h_array[self.__phi_rowindex, self.__phi_columnindex])
        )

        self.__new_h = new_h  # An array of the updated h values for every lattice point of interest (points are listed as start,h-move,phi-move)

    def _update_phi(self):
        """Updates the self.__new_phi attribute, called within make_update():

        - A copy of the self.phi_array is made
        - A series of random height fluctuations are generated, scaled by self.fluct_phi
        - These fluctuations are added to the starting lattice points
        - These fluctuations are subtracted from the corresponding nearest neighbour
        - Additional structure is used to ensure phi is always positive
        - The new phi values in this copied array are saved under the attribute self.__new_phi
        """

        # Making a copy of original phi-array points that can be manipulated freely

        startphi = self.phi_array[
            self.__start_rowindex, self.__start_columnindex
        ].copy()
        movephi = self.phi_array[self.__phi_rowindex, self.__phi_columnindex].copy()

        fluct_phi = self.fluct_phi * (
            random.random(self.n * self.m) - 0.5
        )  # Generating random phi fluctuations

        sum_startphi = startphi + fluct_phi

        sum_movephi = movephi - fluct_phi

        # Ensuring phi is always >=0

        new_startphi = np.where(
            (sum_startphi >= 0) & (sum_movephi >= 0), sum_startphi, startphi
        )

        new_movephi = np.where(
            (sum_startphi >= 0) & (sum_movephi >= 0), sum_movephi, movephi
        )

        new_phi = np.concatenate(
            (
                new_startphi,
                self.phi_array[self.__h_rowindex, self.__h_columnindex],
                new_movephi,
            )
        )

        self.__new_phi = new_phi  # An array of the updated phi values for every lattice point of interest (listed as start,h-move,phi-move)

    def _delta_energy_el(self) -> np.ndarray:
        """Calculates the change in electrostatic energy associated with updating from phi to new_phi, used in self.calc_energy_change():

        - Energy change is calculated for all relevant lattice points in each 4x4 grid i.e. the starting points and the chosen nearest neighbours,
            the energy change of both points is summed into the output array.

        Returns:
            np.ndarray: An array that contains the total electrostatic energy changes in each of the 4x4 grids. shape(n*m)
        """

        el_energy = self.a1 * (
            self.__new_phi ** (5 / 2)
            - self.phi_array[self.__all_rowindex, self.__all_columnindex] ** (5 / 2)
        )
        sum_el_energy = (el_energy.reshape(3, -1)).sum(axis=0)
        return sum_el_energy  # an array with the change in electrostatic energy for each 4x4grid (array of length equal to n*m)

    def _delta_energy_gr1(self) -> np.ndarray:
        """Calculates the change in gravitational energy associated with updating from phi/h to new_phi/new_h, used in self.calc_energy_change():

        - Energy change is calculated for all relevant lattice points in each 4x4 grid i.e. the starting points and the chosen nearest neighbours,
            the energy change of both points is summed into the output array.

        Returns:
            np.ndarray: An array that contains the total gravitational energy changes in each of the 4x4 grids. shape(n*m)
        """

        gr1_energy = self.a2 * (
            (self.__new_phi * self.__new_h)
            - (
                self.phi_array[self.__all_rowindex, self.__all_columnindex]
                * self.h_array[self.__all_rowindex, self.__all_columnindex]
            )
        )

        sum_gr1_energy = (gr1_energy.reshape(3, -1)).sum(axis=0)
        return sum_gr1_energy  # an array with the change in colloid GPE for each 4x4 grid (array of length equal to n*m)

    def _delta_energy_gr2(self) -> np.ndarray:
        """Calculates the change in fluid gravitational energy associated with updating from phi/h to new_phi/new_h, used in self.calc_energy_change():

        - Energy change is calculated for all relevant lattice points in each 4x4 grid i.e. the starting points and the chosen nearest neighbours,
            the energy change of both points is summed into the output array.

        Returns:
            np.ndarray: An array that contains the total electrostatic energy changes in each of the 4x4 grids. shape(n*m)
        """
        # Calcs change in gravity due to h**2 term (not really sure how this works!)
        gr2_energy = -(
            0.5
            * self.a4
            * (
                self.__new_h**2
                - self.h_array[self.__all_rowindex, self.__all_columnindex] ** 2
            )
        )
        sum_gr2_energy = (gr2_energy.reshape(3, -1)).sum(axis=0)

        return sum_gr2_energy  # an array with the change in fluid GPE for each 4x4 grid (array of length equal to n*m)

    def _delta_energy_st(self) -> np.ndarray:
        """Calculates the change in surface tension energy associated with updating from h to new_h, used in self.calc_energy_change():

        - Energy change is calculated for all relevant lattice points in each 4x4 grid i.e. the starting points and the chosen nearest neighbours,
            the energy change of both points is summed into the output array.

        Returns:
            np.ndarray: An array that contains the total surface tension energy changes in each of the 4x4 grids. shape(n*m)
        """

        # This is non-trivial, involves considering an alternating diagonal splitting triangular mesh and calculating the area change of each triangle
        # Area calculations use a first order approx in fluct_h/L where L is the real space distance between lattice points

        new_h_array = self.h_array.copy()
        new_h_array[self.__all_rowindex, self.__all_columnindex] = self.__new_h

        st_energy = np.zeros(int(len(self.__new_h) * 2 / 3))

        # for convenience create a row and column index list of all indices that have an altered height

        changed_rowindex = np.concatenate((self.__start_rowindex, self.__h_rowindex))
        changed_columnindex = np.concatenate(
            (self.__start_columnindex, self.__h_columnindex)
        )

        # Creating a boolean mask that will be used to stop double counting
        mask = np.ones((4 * (self.n + 1), 4 * (self.m + 1))).astype(bool)
        mask[self.__h_rowindex, self.__h_columnindex] = False

        # To cut out repetitiveness
        original_h = self.h_array[changed_rowindex, changed_columnindex]
        updated_h = self.__new_h[
            : (int(len(self.__new_h) * (2 / 3)))
        ]  # Only interested in the lattice points where h has changed

        # Need to 'roll' in 4 directions and also multiply by mask each time:

        st_energy += (
            (
                np.roll(new_h_array, 1, axis=1)[changed_rowindex, changed_columnindex]
                - updated_h
            )
            * np.roll(mask, 1, axis=1)[changed_rowindex, changed_columnindex]
        ) ** 2 - (
            (
                np.roll(self.h_array, 1, axis=1)[changed_rowindex, changed_columnindex]
                - original_h
            )
            * np.roll(mask, 1, axis=1)[changed_rowindex, changed_columnindex]
        ) ** 2

        st_energy += (
            (
                np.roll(new_h_array, -1, axis=1)[changed_rowindex, changed_columnindex]
                - updated_h
            )
            * np.roll(mask, -1, axis=1)[changed_rowindex, changed_columnindex]
        ) ** 2 - (
            (
                np.roll(self.h_array, -1, axis=1)[changed_rowindex, changed_columnindex]
                - original_h
            )
            * np.roll(mask, -1, axis=1)[changed_rowindex, changed_columnindex]
        ) ** 2

        st_energy += (
            (
                np.roll(new_h_array, 1, axis=0)[changed_rowindex, changed_columnindex]
                - updated_h
            )
            * np.roll(mask, 1, axis=0)[changed_rowindex, changed_columnindex]
        ) ** 2 - (
            (
                np.roll(self.h_array, 1, axis=0)[changed_rowindex, changed_columnindex]
                - original_h
            )
            * np.roll(mask, 1, axis=0)[changed_rowindex, changed_columnindex]
        ) ** 2

        st_energy += (
            (
                np.roll(new_h_array, -1, axis=0)[changed_rowindex, changed_columnindex]
                - updated_h
            )
            * np.roll(mask, -1, axis=0)[changed_rowindex, changed_columnindex]
        ) ** 2 - (
            (
                np.roll(self.h_array, -1, axis=0)[changed_rowindex, changed_columnindex]
                - original_h
            )
            * np.roll(mask, -1, axis=0)[changed_rowindex, changed_columnindex]
        ) ** 2

        st_energy *= 0.5 * self.a3
        sum_st_energy = (st_energy.reshape(2, -1)).sum(axis=0)
        return sum_st_energy  # an array with the change in surface tension energy for each 4x4 grid (array of length equal to n*m)

    def _calc_energy_change(self):
        """Calculates the total energy changes associated with updating each 4x4 grid and stores this in the self.energy_change attribute.
        - self.energy_change will be an array of length n*m i.e. an energy change associated with each grid
        - This is called within the make_update() function
        """

        total = np.zeros(self.n * self.m)

        total += self.delta_energy_el()
        total += self.delta_energy_gr1()
        total += self.delta_energy_gr2()
        total += self.delta_energy_st()
        self.__energy_change = total

    def _check_update(self) -> np.ndarray:
        """Uses the Boltzmann/Metropolis criterion to decide whether to update each of the 4x4 grids. Called within make_update().

        Returns:
            np.ndarray: Boolean array of length 3*n*m (1:update, 0:don't update) for all 3*n*m relevant lattice points
        """

        total = self.__energy_change
        parray = np.exp(-self.beta * total)  # Metropolis criterion

        check_array = np.where(
            (np.random.random(self.n * self.m) < parray),
            np.ones(self.n * self.m),
            np.zeros(self.n * self.m),
        )  # 1 means move is accepted, 0 means move is rejected

        return np.tile(
            check_array, 3
        )  # can be used as a mask for updating all 3*n*m squares

    def make_update(self):
        """Completes the full process for one step of the Metropolis algorithm:
        - Chooses new indices i.e. lattice points to alter
        - Chooses new potential phi and h values
        - Calculates energies associated with these updates
        - Determines whether updates are accepted or not and carries out these updates on self.h_array, self.phi_array

        Additional Functionality:
        - For each call of make_update(), one count is added to the iteration_count attribute
        - For each call of make_update(), all the accepted energy changes are summed and added to the energy_count attribute
        """
        self._update_move_indices()
        self._update_h()
        self._update_phi()
        self._calc_energy_change()
        update_array = self._check_update()

        self.h_array[self.__all_rowindex, self.__all_columnindex] = np.where(
            update_array == 1,
            self.__new_h,
            self.h_array[self.__all_rowindex, self.__all_columnindex],
        )

        self.phi_array[self.__all_rowindex, self.__all_columnindex] = np.where(
            update_array,
            self.__new_phi,
            self.phi_array[self.__all_rowindex, self.__all_columnindex],
        )
        self.iteration_count += 1
        self.energy_count += np.sum(
            (update_array[: self.n * self.m] * self.__energy_change)
        )

    def save_lattice(self, filename):
        """Saves the current instance of the Lattice class using pickle
        Args:
            filename (str): Name of saved file, must end in .pkl
        """
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        """Class method used to load a previously saved class instance e.g. new_instance=Lattice.load('filenmae.pkl')
        Args:
            filename (str): Name of saved file, must end in .pkl"""

        with open(filename, "rb") as f:
            return pickle.load(f)


if __name__ == "__main__":

    """-----------------------------------------------------------------------------------------

    This is some example use of the Lattice class"""

    newgrid = Lattice(
        5, 5
    )  # Generates a lattice with a certain size, all other parameters could also be explicitly specified here
    # e.g. newgrid.beta=1, newgrid.a1=10

    interval = 1000
    iterations = 500000

    # Creates empty arrays to store quantites of interest
    energy_array = np.zeros(iterations)[::interval]
    phi_std_array = np.zeros(iterations)[::interval]

    for i in range(iterations):

        # Could edit any of the parameters during the iteration process
        # e.g. could dynamically alter beta using a Simulated Annealing algorithm
        # (newgrid.beta = 0.0001 + 0.2069 * np.log(1 + i))

        newgrid.make_update()  # repetitively updating the lattice

        if (
            i % interval == 0
        ):  # periodically recording the energy changes/any other quantities of interest
            newgrid.update_stats()  # have to explicitly call update_stats() isn't included in make_update()
            energy_array[i // interval] = newgrid.energy_count
            phi_std_array[i // interval] = newgrid.phi_std
            print(f"Iteration {i} Completed")

    # Plotting results

    newgrid.draw_fields(save=False)
    # print(f"Mean h:{newgrid.h_mean}\nMean phi:{newgrid.phi_mean}\nStd. Phi: {newgrid.phi_std}")

    fig, ax1 = plt.subplots()

    color1 = "tab:blue"
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Energy", color=color1)
    ax1.plot(np.arange(iterations)[::interval], energy_array, color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()

    color2 = "tab:green"
    ax2.set_ylabel("Deviation of phi", color=color2)
    ax2.plot(np.arange(iterations)[::interval], phi_std_array, color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    fig.suptitle("Total energy and Deviation of Phi Disturbution")
    plt.show()
