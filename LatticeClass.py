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
        initial_phi=0.001,  # Initial magnitude of uniform phi distribution
        fluct_phi=5,  # Max. percentage for phi fluctuations at each step (+- fluct_phi%)
        fluct_h=5,  # Max. percentage change of h
        min_fluct_h=0.0001,  # Min. size for a h fluctuation
        a1=12,  # coefficient of electrostatic interactions (repulsion)
        a2=13,  # coefficient of extra density of colloids
        a3=2000,  # coeff of surface tension
        a4=1.3,  # coefficient of density difference of liquids
        beta=200000,
    ) -> None:

        # Counters for the number of make_update() iterations and total energy change of system

        self.energy_count = 0
        self.energy_array = []
        self.iteration_count = 0

        # Initialising coefficients/parameters as modifiable class attributes
        self.n = n
        self.m = m

        self.initial_phi = initial_phi
        self.fluct_h = fluct_h
        self.fluct_phi = fluct_phi
        self.min_fluct_h = min_fluct_h

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

        self.__new_h = np.zeros(2 * self.m * self.n)
        self.__new_phi = np.zeros(2 * self.m * self.n)
        self.__energy_change = np.zeros(self.m * self.n)

        self.__h_update_rowindex, self.__h_update_columnindex = np.zeros(
            2 * self.m * self.n
        ), np.zeros(2 * self.m * self.n)
        self.__phi_update_rowindex, self.__phi_update_columnindex = np.zeros(
            2 * self.m * self.n
        ), np.zeros(2 * m * n)
        self.__update_rowindex, self.__update_columnindex = np.zeros(
            2 * m * n
        ), np.zeros(2 * m * n)

        # Initialises attributes that track mean and std of h and phi arrays
        self.h_mean, self.h_std = np.mean(self.h_array), np.std(self.h_array)
        self.phi_mean, self.phi_std = np.mean(self.phi_array), np.std(self.phi_array)
        self.phi_std_array = []

    def calc_absolute_energy(self):
        """Calculates the energy of the system relative to a 0 point corresponding to a uniform phi array and h=0"""
        gr1_energy = self.a2 * (self.h_array * self.phi_array).sum()
        gr2_energy = self.a4 * (self.h_array**2).sum()

        # The other two energy terms have more involved calculations
        uniform_phi = np.mean(self.phi_array) * np.ones_like(self.phi_array)
        el_energy = (
            self.a1 * (self.phi_array ** (5 / 2) - uniform_phi ** (5 / 2))
        ).sum()

        # This is non-trivial, involves considering an alternating diagonal splitting triangular mesh and calculating the area change of each triangle
        # Area calculations use a first order approx in fluct_h/L where L is the real space distance between lattice points

        mask = np.pad(
            np.ones_like(self.h_array), pad_width=1, mode="constant", constant_values=0
        )

        # Create a checkerboard pattern of indices
        checkerboard = np.indices(self.h_array.shape).sum(axis=0) % 2 == 0
        padded_checkerboard = np.pad(
            checkerboard, pad_width=1, mode="constant", constant_values=False
        )

        h0_array = self.h_array[checkerboard]
        st_energy = np.zeros_like(h0_array)
        st_energy += (
            (np.roll(self.h_array, 1, axis=1)[checkerboard] - h0_array)
            * np.roll(mask, 1, axis=1)[padded_checkerboard]
        ) ** 2

        st_energy += (
            (np.roll(self.h_array, -1, axis=1)[checkerboard] - h0_array)
            * np.roll(mask, -1, axis=1)[padded_checkerboard]
        ) ** 2

        st_energy += (
            (np.roll(self.h_array, 1, axis=0)[checkerboard] - h0_array)
            * np.roll(mask, 1, axis=0)[padded_checkerboard]
        ) ** 2

        st_energy += (
            (np.roll(self.h_array, -1, axis=0)[checkerboard] - h0_array)
            * np.roll(mask, -1, axis=0)[padded_checkerboard]
        ) ** 2

        st_energy /= 2
        st_energy *= self.a3
        st_energy = st_energy.sum()

        return el_energy + gr1_energy + gr2_energy + st_energy

    def set_initial_conditions(self):
        """- Sets initial energy counter (with 0 corresponding to a uniform phi array and h=0)
        - Initialises arrays for tracking energy and phi standard deviation
        - Sets iteration count to 0
        """

        self.energy_count = self.calc_absolute_energy()

        self.iteration_count = 0
        self.energy_array = []
        self.phi_std_array = []

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
            self.update_all_stats()

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

    def update_all_stats(self):
        """Updates the height and phi mean and standard deviation attributes.
        [To save computation time, this method is not included in every call of make_update()]
        """

        self.h_mean, self.h_std = np.mean(self.h_array), np.std(self.h_array)
        self.phi_mean, self.phi_std = np.mean(self.phi_array), np.std(self.phi_array)

    def _track_phi_std(self):
        """Appends the current standard deviation of the phi array to the phi_std_array attribute.
        [To save computation time, this method is only called every 1000 iterations]"""
        self.phi_std = np.std(self.phi_array)
        self.phi_std_array.append(np.std(self.phi_array))

    def _update_move_indices(self):
        """Chooses which lattice points to try and alter for both h and phi, this is called within make_update():

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

        self.__h_update_rowindex = np.concatenate(
            (self.__start_rowindex, self.__h_rowindex)
        ).astype(np.int32)

        self.__h_update_columnindex = np.concatenate(
            (self.__start_columnindex, self.__h_columnindex)
        ).astype(np.int32)

        self.__phi_update_rowindex = np.concatenate(
            (self.__start_rowindex, self.__phi_rowindex)
        ).astype(np.int32)

        self.__phi_update_columnindex = np.concatenate(
            (self.__start_columnindex, self.__phi_columnindex)
        ).astype(np.int32)

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

    def _set_h_update(self):
        """Sets the new_h,new_phi and update_indices values for an h update; called within make_update().

        - Sets the update indices to those associated with an h update

        - A copy of the (relevant) intial h values is made
        - h fluctations are: U*fluct_h/100*(|h|+(min_fluct_h*100/fluct_h)) (U is from random uniform -1:1)
        - Negative fluctations are scaled by h at start position
        - Positive fluctuations are scaled by h at end position
        - These fluctuations are added/subtracted to the initial/new h values
        - The new h values are saved under the attribute self.__new_h

        - new_phi is trivially set to the current phi values at the update indices associated with the h update
        """

        # Ensures the update indices are associated with an h change
        self.__update_rowindex = self.__h_update_rowindex
        self.__update_columnindex = self.__h_update_columnindex

        starth = self.h_array[self.__start_rowindex, self.__start_columnindex].copy()

        moveh = self.h_array[self.__h_rowindex, self.__h_columnindex].copy()

        fluctuations = (self.fluct_h / 100) * 2 * (random.random(self.n * self.m) - 0.5)

        fluct_constant = self.min_fluct_h * (100 / self.fluct_h)

        scaled_fluctuations = np.where(
            (fluctuations >= 0),
            fluctuations * (np.abs(moveh) + fluct_constant),
            fluctuations * (np.abs(starth) + fluct_constant),
        )

        starth = starth + scaled_fluctuations
        moveh = moveh - scaled_fluctuations

        new_h = np.concatenate(
            (
                starth,
                moveh,
            )
        )

        # Sets the h and phi values of the updated lattice points
        self.__new_h = new_h
        self.__new_phi = self.phi_array[
            self.__update_rowindex, self.__update_columnindex
        ]

    def _set_phi_update(self):
        """Sets the new_h,new_phi and update_indices values for a phi update; called within make_update().

         - Sets the update indices to those associated with an h update

        - A copy of the (relevant) initial phi values is made
        - A series of random fluctuations are made [-1,1] and scaled by self.fluct_phi %
        - Negative fluctations are scaled by phi at start position
        - Positive fluctuations are scaled by phi at end position
        - Fluctuations are added and subtracted to relevant lattice points
        - The new phi values in this copied array are saved under the attribute self.__new_phi

        - new_h is trivially set to the current h values at the update indices associated with the phi update


        """

        # Ensures the update indices are associated with a phi change
        self.__update_rowindex = self.__phi_update_rowindex
        self.__update_columnindex = self.__phi_update_columnindex

        startphi = self.phi_array[
            self.__start_rowindex, self.__start_columnindex
        ].copy()

        movephi = self.phi_array[self.__phi_rowindex, self.__phi_columnindex].copy()

        fluctuations = (
            (self.fluct_phi / 100) * 2 * (random.random(self.n * self.m) - 0.5)
        )

        scaled_fluctuations = np.where(
            (fluctuations >= 0), fluctuations * movephi, fluctuations * startphi
        )

        startphi = startphi + scaled_fluctuations
        movephi = movephi - scaled_fluctuations

        new_phi = np.concatenate(
            (
                startphi,
                movephi,
            )
        )

        # Sets the h and phi values of the updated lattice points
        self.__new_phi = new_phi
        self.__new_h = self.h_array[self.__update_rowindex, self.__update_columnindex]

    def _delta_energy_el(self) -> np.ndarray:
        """Calculates the change in electrostatic energy associated with updating from phi to new_phi, used in self._calc_energy_change_phi():

        - Energy change is calculated for all relevant lattice points in each 4x4 grid i.e. the starting points and the chosen nearest neighbours,
            the energy change of both points is summed into the output array.

        Returns:
            np.ndarray: An array that contains the total electrostatic energy changes in each of the 4x4 grids. shape(n*m)
        """

        el_energy = self.a1 * (
            self.__new_phi ** (5 / 2)
            - self.phi_array[self.__update_rowindex, self.__update_columnindex]
            ** (5 / 2)
        )
        sum_el_energy = (el_energy.reshape(2, -1)).sum(axis=0)
        return sum_el_energy  # an array with the change in electrostatic energy for each 4x4grid (array of length equal to n*m)

    def _delta_energy_gr1(self) -> np.ndarray:
        """Calculates the change in gravitational energy associated with updating from phi/h to new_phi/new_h, used in self._calc_energy_change_phi() and self._calc_energy_change_h():

        - Energy change is calculated for all relevant lattice points in each 4x4 grid i.e. the starting points and the chosen nearest neighbours,
            the energy change of both points is summed into the output array.

        Returns:
            np.ndarray: An array that contains the total gravitational energy changes in each of the 4x4 grids. shape(n*m)
        """

        gr1_energy = self.a2 * (
            (self.__new_phi * self.__new_h)
            - (
                self.phi_array[self.__update_rowindex, self.__update_columnindex]
                * self.h_array[self.__update_rowindex, self.__update_columnindex]
            )
        )

        sum_gr1_energy = (gr1_energy.reshape(2, -1)).sum(axis=0)
        return sum_gr1_energy  # an array with the change in colloid GPE for each 4x4 grid (array of length equal to n*m)

    def _delta_energy_gr2(self) -> np.ndarray:
        """Calculates the change in fluid gravitational energy associated with updating from h to new_h, used in self._calc_energy_change_h():

        - Energy change is calculated for all relevant lattice points in each 4x4 grid i.e. the starting points and the chosen nearest neighbours,
            the energy change of both points is summed into the output array.

        Returns:
            np.ndarray: An array that contains the total electrostatic energy changes in each of the 4x4 grids. shape(n*m)
        """
        # Calcs change in gravity due to h**2 term (not really sure how this works!)
        gr2_energy = self.a4 * (
            self.__new_h**2
            - self.h_array[self.__update_rowindex, self.__update_columnindex] ** 2
        )

        sum_gr2_energy = (gr2_energy.reshape(2, -1)).sum(axis=0)

        return sum_gr2_energy  # an array with the change in fluid GPE for each 4x4 grid (array of length equal to n*m)

    def _delta_energy_st(self) -> np.ndarray:
        """Calculates the change in surface tension energy associated with updating from h to new_h, used in self._calc_energy_change_h():

        - Energy change is calculated for all relevant lattice points in each 4x4 grid i.e. the starting points and the chosen nearest neighbours,
            the energy change of both points is summed into the output array.

        Returns:
            np.ndarray: An array that contains the total surface tension energy changes in each of the 4x4 grids. shape(n*m)
        """

        # This is non-trivial, involves considering an alternating diagonal splitting triangular mesh and calculating the area change of each triangle
        # Area calculations use a first order approx in fluct_h/L where L is the real space distance between lattice points

        new_h_array = self.h_array.copy()
        new_h_array[self.__update_rowindex, self.__update_columnindex] = self.__new_h

        st_energy = np.zeros(int(len(self.__new_h)))

        # For convenience/readibility

        changed_rowindex = self.__update_rowindex
        changed_columnindex = self.__update_columnindex

        # Creating a boolean mask that will be used to stop double counting
        mask = np.ones((4 * (self.n + 1), 4 * (self.m + 1))).astype(bool)
        mask[self.__h_rowindex, self.__h_columnindex] = False

        # To cut out repetitiveness
        original_h = self.h_array[changed_rowindex, changed_columnindex]
        updated_h = self.__new_h
        # Only interested in the lattice points where h has changed

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

    def _calc_energy_change_h(self):
        """Calculates the total energy changes associated with updating h only in each 4x4 grid and stores this in the self.energy_change_h attribute.
        - self.__energy_change_h will be an array of length n*m i.e. an energy change associated with each grid
        - This is called within the make_update() function
        """

        total = np.zeros(self.n * self.m)

        total += self._delta_energy_gr1()
        total += self._delta_energy_gr2()
        total += self._delta_energy_st()
        self.__energy_change = total

    def _calc_energy_change_phi(self):
        """Calculates the total energy changes associated with updating phi only in each 4x4 grid and stores this in the self.energy_change_h attribute.
        - self.__energy_change_phi will be an array of length n*m i.e. an energy change associated with each grid
        - This is called within the _make_update() function
        """

        total = np.zeros(self.n * self.m)

        total += self._delta_energy_el()
        total += self._delta_energy_gr1()
        self.__energy_change = total

    def _check_update_h(self) -> np.ndarray:
        """Uses the Boltzmann/Metropolis criterion to decide whether to update h for each of the 4x4 grids. Called within make_update().

        Returns:
            np.ndarray: Boolean array of length 2*n*m (1:update, 0:don't update) for all 2*n*m relevant lattice points
        """
        # |h|+(min_fluct_h*100/fluct_h)
        total = self.__energy_change
        parray = (
            np.exp(-self.beta * total)
            * (
                np.abs(self.h_array[self.__start_rowindex, self.__start_columnindex])
                + self.min_fluct_h * 100 / self.fluct_h
            )
            / (
                np.abs(self.h_array[self.__h_rowindex, self.__h_columnindex])
                + self.min_fluct_h * 100 / self.fluct_h
            )
        )  # Metropolis criterion

        check_array = np.where(
            (np.random.random(self.n * self.m) < parray),
            np.ones(self.n * self.m),
            np.zeros(self.n * self.m),
        )  # 1 means move is accepted, 0 means move is rejected

        return np.tile(
            check_array, 2
        )  # can be used as a mask for updating all 2*n*m relevant lattice points

    def _check_update_phi(self) -> np.ndarray:
        """Uses the Boltzmann/Metropolis criterion (with added factor due to phi fluctuations being proportional to phi) to decide whether to update phi for each of the 4x4 grids. Called within make_update().

        Returns:
            np.ndarray: Boolean array of length 2*n*m (1:update, 0:don't update) for all 2*n*m relevant lattice points
        """

        total = self.__energy_change
        parray = (
            np.exp(-self.beta * total)
            * self.phi_array[self.__start_rowindex, self.__start_columnindex]
            / self.phi_array[self.__phi_rowindex, self.__phi_columnindex]
        )  # Metropolis criterion

        check_array = np.where(
            (np.random.random(self.n * self.m) < parray),
            np.ones(self.n * self.m),
            np.zeros(self.n * self.m),
        )  # 1 means move is accepted, 0 means move is rejected

        return np.tile(
            check_array, 2
        )  # can be used as a mask for updating all 2*n*m relevant lattice points

    def make_update(self):
        """Completes the full process for one step of the Metropolis algorithm:
        - Chooses new indices i.e. lattice points to alter
        - First considers h fluctuations
        - Calculates energies associated with these fluctuations
        - Determines whether fluctations are accepted or not and carries out these updates on self.h_array
        - Adds energy change to energy_count attribute
        - Repeats process for phi
        - Adds one to the iteration_count attribute

        Additional Functionality:
        - For each call of make_update(), one count is added to the iteration_count attribute
        - For each call of make_update(), all the accepted energy changes are summed and added to the energy_count attribute
        """
        self._update_move_indices()

        # Perform h update
        self._set_h_update()

        self._calc_energy_change_h()

        update_array = self._check_update_h()

        self.h_array[self.__update_rowindex, self.__update_columnindex] = np.where(
            update_array == 1,
            self.__new_h,
            self.h_array[self.__update_rowindex, self.__update_columnindex],
        )
        self.energy_count += np.sum(
            (update_array[: self.n * self.m] * self.__energy_change)
        )

        # Repeat process for phi

        self._set_phi_update()

        self._calc_energy_change_phi()

        update_array = self._check_update_phi()

        self.phi_array[self.__update_rowindex, self.__update_columnindex] = np.where(
            update_array == 1,
            self.__new_phi,
            self.phi_array[self.__update_rowindex, self.__update_columnindex],
        )
        self.energy_count += np.sum(
            (update_array[: self.n * self.m] * self.__energy_change)
        )

        self.iteration_count += 1
        self.energy_array.append(self.energy_count)
        if self.iteration_count % 1000 == 0:
            self._track_phi_std()

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
        15, 15
    )  # Generates a lattice with a certain size, all other parameters could also be explicitly specified here
    # e.g. newgrid.beta=1, newgrid.a1=10

    interval = 1000
    iterations = 75000

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

    fig.suptitle("Total Energy and Deviation of Phi Disturbution")
    plt.show()
