"""Tweaking the original Lattice class to work with multiple trials simulataneously by using a 3D lattice array
- This will hopefully be transferrable to CuPy and GPU acceleration"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.random as random

# Used for saving data

import pickle


class Test3D:
    def __init__(self, dimension: int, num_trials: int) -> None:

        # Initialise parameters for each trial with default values

        self.num_trials = num_trials

        self.a1 = np.ones(self.num_trials) * 12
        self.a2 = np.ones(self.num_trials) * 13
        self.a3 = np.ones(self.num_trials) * 2000
        self.a4 = np.ones(self.num_trials) * 1.3
        self.a5 = np.ones(self.num_trials) * 20000

        self.beta = np.ones(self.num_trials) * 20000

        self.initial_phi = (
            np.ones(self.num_trials) * 0.001
        )  # Mean concentration (colloids per lattice point)

        # These will most likely be consistent across all trials

        self.fluct_phi = (
            5  # Max. percentage for phi fluctuations at each step (+- fluct_phi%)
        )
        self.fluct_h = 5  # Max. percentage change of h
        self.min_fluct_h = 0.0001  # Min. size for a h fluctuation
        self.dimension = dimension  # Num of 4x4 grids per row/column

        # Initialise array to store data
        """Each array has size (dimension x dimension x num_trials) 
        - h values are set uniformly to 0
        - phi values are set uniformly to initial_phi for each trial
        """
        self.h_array = np.zeros(
            (self.num_trials, 4 * (self.dimension + 1), 4 * (self.dimension + 1))
        )
        self.phi_array = np.empty(
            (self.num_trials, 4 * (self.dimension + 1), 4 * (self.dimension + 1))
        )
        self.phi_array[:, :, :] = self.initial_phi[:, np.newaxis, np.newaxis]

        self.__update_h_array = self.h_array.copy()
        self.__update_phi_array = (
            self.phi_array.copy()
        )  # Array to store updated values before they are accepted

        # Initialise 1D coordinate arrays to locate start and move points

        self.__startcoords = np.empty(self.num_trials * self.dimension**2)

        self.__movecoords = np.empty(self.num_trials * self.dimension**2)

        # Initialise 1D arrays that will store the action of the coords on the h and phi arrays

        self.__start_h = np.empty(self.num_trials * self.dimension**2)
        self.__move_h = np.empty(self.num_trials * self.dimension**2)
        self.__start_phi = np.empty(self.num_trials * self.dimension**2)
        self.__move_phi = np.empty(self.num_trials * self.dimension**2)

        self.__fluctstart_phi = np.empty(self.num_trials * self.dimension**2)
        self.__fluctmove_phi = np.empty(self.num_trials * self.dimension**2)

        self.__fluctstart_h = np.empty(self.num_trials * self.dimension**2)
        self.__fluctmove_h = np.empty(self.num_trials * self.dimension**2)

        # Arrays to store energy of each trial
        self.energy_count = np.zeros(self.num_trials)
        self.energy_array = []

        # Arrays to store iterations and other properties of interest
        self.iteration_count = 0
        self.phi_std, self.phi_mean = np.zeros(self.num_trials), np.zeros(
            self.num_trials
        )
        self.h_mean, self.h_std = np.zeros(self.num_trials), np.zeros(self.num_trials)

        self.phi_std_array = []

    def calc_absolute_energy(self):
        """Calculates the energy of each trial with a 0 point corresponding to uniform phi=initial_phi and h=0
        - This method is more costly than just calculating the energy change at each step so must be called explicitly

        Returns: 1D np.array of length num_trials containing the 'absolute' energy of each trial
        """

        gr1_energy = self.a2 * np.sum((self.phi_array * self.h_array), axis=(1, 2))
        gr2_energy = self.a4 * np.sum(self.h_array**2, axis=(1, 2))

        # Calc. electrostatic energy relative to a uniform phi array
        uniform_phi = np.mean(self.phi_array, axis=(1, 2))[
            :, np.newaxis, np.newaxis
        ] * np.ones((1, 4 * (self.dimension + 1), 4 * (self.dimension + 1)))

        el_energy = self.a1 * np.sum(
            (self.phi_array ** (5 / 2) - uniform_phi**5 / 2), axis=(1, 2)
        )

        # This is non-trivial, involves considering an alternating diagonal splitting triangular mesh and calculating the area change of each triangle
        # Area calculations use a first order approx in fluct_h/L where L is the real space distance between lattice points
        """Note this could be improved for GPU use by using 1D coordinates rather than boolean masks"""
        mask = np.pad(
            np.ones((4 * (self.dimension + 1), 4 * (self.dimension + 1))),
            pad_width=1,
            mode="constant",
            constant_values=0,
        )
        # Duplicate the mask to have num_trials copies
        mask_3D = np.repeat(mask[np.newaxis, :, :], self.num_trials, axis=0)

        # Create a checkerboard pattern of indices
        checkerboard = (
            np.indices((4 * (self.dimension + 1), 4 * (self.dimension + 1))).sum(axis=0)
            % 2
            == 0
        )
        checkerboard_3D = np.repeat(
            checkerboard[np.newaxis, :, :], self.num_trials, axis=0
        )
        padded_checkerboard = np.pad(
            checkerboard, pad_width=1, mode="constant", constant_values=False
        )
        padded_checkerboard_3D = np.repeat(
            padded_checkerboard[np.newaxis, :, :], self.num_trials, axis=0
        )

        h0_array = self.h_array[
            checkerboard_3D
        ]  # This is a 1D array ordered as [all checkerboard points in trial i],[all checkerboard points in trial i+1],...

        st_energy = (
            (np.roll(self.h_array, 1, axis=1)[checkerboard_3D] - h0_array)
            * np.roll(mask_3D, 1, axis=1)[padded_checkerboard_3D]
        ) ** 2
        st_energy += (
            (np.roll(self.h_array, -1, axis=1)[checkerboard_3D] - h0_array)
            * np.roll(mask_3D, -1, axis=1)[padded_checkerboard_3D]
        ) ** 2
        st_energy += (
            (np.roll(self.h_array, 1, axis=2)[checkerboard_3D] - h0_array)
            * np.roll(mask_3D, 1, axis=2)[padded_checkerboard_3D]
        ) ** 2
        st_energy += (
            (np.roll(self.h_array, -1, axis=2)[checkerboard_3D] - h0_array)
            * np.roll(mask_3D, -1, axis=2)[padded_checkerboard_3D]
        ) ** 2

        # st_energy is a 1D array of length (checkerboard points in one trial)*num_trials
        # need to condense it to a 1D array of length num_trials

        st_energy = st_energy.reshape(self.num_trials, -1).sum(axis=1)

        st_energy = 0.5 * self.a3 * np.sum(st_energy, axis=(1, 2))

        return gr1_energy + gr2_energy + el_energy + st_energy

    def set_initial_conditions(self):
        """- (Re)Sets initial energy and iteration counter
        - (Re)Sets energy and phi std arrays
        Should call this explicitly before running a test
        """
        self.energy_count = self.calc_absolute_energy()
        self.phi_std_array = []
        self.iteration_count = 0
        self.energy_array = []

    def draw_fields(self, trial_num, save=False, name=None):
        """Plots the phi and h fields for a given trial
        Gives option to save the plots with a name"""

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
            plt.savefig(f"{name}.png", dpi=300)

        plt.show()

    def update_all_stats(self):
        """Updates the height and phi mean and standard deviation attributes.
        [To save computation time, this method is not included in every call of make_update()]
        """

        self.h_mean, self.h_std = np.mean(self.h_array, axis=(1, 2)), np.std(
            self.h_array, axis=(1, 2)
        )
        self.phi_mean, self.phi_std = np.mean(self.phi_array, axis=(1, 2)), np.std(
            self.phi_array, axis=(1, 2)
        )

    def _choose_indices(self):
        """- Chooses random start point within 4x4 tile (same for each tile in the same 2D lattice but differs for different trials)
        - Chooses random move point within 4x4 tile (different for every tile and trial)
        - Stores results in the form of 1D coordinates to be used on flattened arrays with np.take()
        """
        x_offset = np.random.randint(0, 4, self.num_trials)
        y_offset = np.random.randint(0, 4, self.num_trials)
        pad = 2

        coords = (
            np.repeat(np.arange(self.num_trials), (self.dimension) ** 2)
            * (4 * (self.dimension + 1)) ** 2,
        )

        coords += (
            (
                np.tile(
                    np.repeat(np.arange(self.dimension) * 4 + pad, self.dimension),
                    self.num_trials,
                )
                + np.repeat(y_offset, self.dimension**2)
            )
            * 4
            * (self.dimension + 1)
        )

        coords += np.tile(
            np.tile(np.arange(self.dimension) * 4 + pad, self.dimension),
            self.num_trials,
        ) + np.repeat(x_offset, self.dimension**2)

        move_choices = [
            (0, 1),
            (0, -1),
            (1, 0),
            (-1, 0),
            (1, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
        ]
        random_moves = np.array(move_choices)[
            np.random.choice(
                len(move_choices), (self.dimension * self.dimension * self.num_trials)
            )
        ]
        move_coords = (
            coords + random_moves[:, 0] * 4 * (self.dimension + 1) + random_moves[:, 1]
        )

        self.__startcoords = coords
        self.__movecoords = move_coords

        # 1D arrays of all relevant h and phi values

        self.__start_h = np.take(self.h_array.flatten(), self.__startcoords)
        self.__move_h = np.take(self.h_array.flatten(), self.__movecoords)
        self.__start_phi = np.take(self.phi_array.flatten(), self.__startcoords)
        self.__move_phi = np.take(self.phi_array.flatten(), self.__movecoords)

    def _create_phi_fluctuations(self):
        """Creates random phi fluctuations and then applies them to the relevant lattice points:

        - A series of random fluctuations are made [-1,1] and scaled by self.fluct_phi %
        - Negative fluctations are scaled by phi at start position
        - Positive fluctuations are scaled by phi at end position
        - Fluctuations are added and subtracted to relevant lattice points on the self.update_array
        """
        self.__update_phi_array = self.phi_array.copy()
        # 1D arrays of all relevant phi values

        fluctuations = (
            random.uniform(-1, 1, self.num_trials * self.dimension**2)
            * self.fluct_phi
            / 100
        )

        scaled_fluctuations = np.where(
            (fluctuations >= 0),
            fluctuations * self.__move_phi,
            fluctuations * self.__start_phi,
        )
        self.__fluctstart_phi = self.__start_phi + scaled_fluctuations
        self.__fluctmove_phi = self.__move_phi - scaled_fluctuations
        self.__fluctstart_h = self.__start_h
        self.__fluctmove_h = self.__move_h

        # Adding fluctuations to the update array
        np.reshape(self.__update_phi_array, -1)[
            self.__startcoords
        ] = self.__fluctstart_phi
        np.reshape(self.__update_phi_array.flatten(), -1)[
            self.__movecoords
        ] = self.__fluctmove_phi

    def _create_h_fluctuations(self):
        """Creates random h fluctuations and then applies them to the relevant lattice points:

        - h fluctations are: U*fluct_h/100*(|h|+(min_fluct_h*100/fluct_h)) (U is from random uniform -1:1)
        - Negative fluctations are scaled by h at start position
        - Positive fluctuations are scaled by h at end position
        - Fluctuations are added and subtracted to relevant lattice points on the self.update_array
        """

        self.__update_h_array = self.h_array.copy()

        fluctuations = (
            random.uniform(-1, 1, self.dimension**2 * self.num_trials)
            * self.fluct_h
            / 100
        )
        fluct_constant = self.min_fluct_h * (100 / self.fluct_h)

        scaled_fluctuations = np.where(
            (fluctuations >= 0),
            fluctuations * (np.abs(self.__move_h) + fluct_constant),
            fluctuations * (np.abs(self.__start_h) + fluct_constant),
        )
        self.__fluctstart_h = self.__start_h + scaled_fluctuations
        self.__fluctmove_h = self.__move_h - scaled_fluctuations
        self.__fluctstart_phi = self.__start_phi
        self.__fluctmove_phi = self.__move_phi

        # Adding fluctuations to the update array
        np.reshape(self.__update_h_array, -1)[self.__startcoords] = self.__fluctstart_h
        np.reshape(self.__update_h_array.flatten(), -1)[
            self.__movecoords
        ] = self.__fluctmove_h

    def _calc_phi_energy_change(self) -> np.ndarray:
        """Calculates the energy change for a phi update.
        Will return an energy change associated with each tile for each trial i.e. a 1D array of size (dimension*dimension)*num_trials
        """
        # Calculating colloid gravitational energy change
        gr1_energy_change = np.repeat(self.a2, self.dimension**2) * (
            self.__fluctstart_phi * self.__fluctstart_h
            - self.__start_phi * self.__start_h
        )
        gr1_energy_change += np.repeat(self.a2, self.dimension**2) * (
            self.__fluctmove_phi * self.__fluctmove_h - self.__move_phi * self.__move_h
        )

        # Calculating electrostatic energy change
        el_energy_change = np.repeat(self.a1, self.dimension**2) * (
            self.__fluctstart_phi ** (5 / 2) - self.__start_phi ** (5 / 2)
        )
        el_energy_change += np.repeat(self.a1, self.dimension**2) * (
            self.__fluctmove_phi ** (5 / 2) - self.__move_phi ** (5 / 2)
        )

        return gr1_energy_change + el_energy_change

    def _calc_h_energy_change(self):
        """Calculates the energy change for an h update.
        Will return an energy change associated with each tile for each trial i.e. a 1D array of size (dimension*dimension)*num_trials
        """
        # Calculating colloid gravitational energy change
        gr1_energy_change = np.repeat(self.a2, self.dimension**2) * (
            self.__fluctstart_phi * self.__fluctstart_h
            - self.__start_phi * self.__start_h
        )
        gr1_energy_change += np.repeat(self.a2, self.dimension**2) * (
            self.__fluctmove_phi * self.__fluctmove_h - self.__move_phi * self.__move_h
        )

        # Calculating gravitational energy change of fluid
        gr2_energy_change = np.repeat(self.a4, self.dimension**2) * (
            self.__fluctstart_h**2 - self.__start_h**2
        )
        gr2_energy_change += np.repeat(self.a4, self.dimension**2) * (
            self.__fluctmove_h**2 - self.__move_h**2
        )

        # Calculating surface tension energy change, this part is not super efficient

        mask = np.ones_like(self.h_array)
        np.reshape(mask, -1)[self.__movecoords] = 0

        # These are all 1D arrays of length (start points per trial)*num_trials

        st_energy = (
            (
                np.take(
                    np.reshape(np.roll(self.__update_h_array, 1, axis=1), -1),
                    self.__startcoords,
                )
                - self.__fluctstart_h
            )
            ** 2
            - (
                np.take(
                    np.reshape(np.roll(self.h_array, 1, axis=1), -1),
                    self.__startcoords,
                )
                - self.__start_h
            )
            ** 2
        ) * np.take(np.reshape(np.roll(mask, 1, axis=1), -1), self.__startcoords)

        st_energy += (
            (
                np.take(
                    np.reshape(np.roll(self.__update_h_array, 1, axis=1), -1),
                    self.__movecoords,
                )
                - self.__fluctmove_h
            )
            ** 2
            - (
                np.take(
                    np.reshape(np.roll(self.h_array, 1, axis=1), -1),
                    self.__movecoords,
                )
                - self.__move_h
            )
            ** 2
        ) * np.take(np.reshape(np.roll(mask, 1, axis=1), -1), self.__movecoords)

        st_energy += (
            (
                np.take(
                    np.reshape(np.roll(self.__update_h_array, -1, axis=1), -1),
                    self.__startcoords,
                )
                - self.__fluctstart_h
            )
            ** 2
            - (
                np.take(
                    np.reshape(np.roll(self.h_array, -1, axis=1), -1),
                    self.__startcoords,
                )
                - self.__start_h
            )
            ** 2
        ) * np.take(np.reshape(np.roll(mask, -1, axis=1), -1), self.__startcoords)

        st_energy += (
            (
                np.take(
                    np.reshape(np.roll(self.__update_h_array, -1, axis=1), -1),
                    self.__movecoords,
                )
                - self.__fluctmove_h
            )
            ** 2
            - (
                np.take(
                    np.reshape(np.roll(self.h_array, -1, axis=1), -1),
                    self.__movecoords,
                )
                - self.__move_h
            )
            ** 2
        ) * np.take(np.reshape(np.roll(mask, -1, axis=1), -1), self.__movecoords)

        st_energy += (
            (
                np.take(
                    np.reshape(np.roll(self.__update_h_array, 1, axis=2), -1),
                    self.__startcoords,
                )
                - self.__fluctstart_h
            )
            ** 2
            - (
                np.take(
                    np.reshape(np.roll(self.h_array, 1, axis=2), -1),
                    self.__startcoords,
                )
                - self.__start_h
            )
            ** 2
        ) * np.take(np.reshape(np.roll(mask, 1, axis=2), -1), self.__startcoords)

        st_energy += (
            (
                np.take(
                    np.reshape(np.roll(self.__update_h_array, 1, axis=2), -1),
                    self.__movecoords,
                )
                - self.__fluctmove_h
            )
            ** 2
            - (
                np.take(
                    np.reshape(np.roll(self.h_array, 1, axis=2), -1),
                    self.__movecoords,
                )
                - self.__move_h
            )
            ** 2
        ) * np.take(np.reshape(np.roll(mask, 1, axis=2), -1), self.__movecoords)

        st_energy += (
            (
                np.take(
                    np.reshape(np.roll(self.__update_h_array, -1, axis=2), -1),
                    self.__startcoords,
                )
                - self.__fluctstart_h
            )
            ** 2
            - (
                np.take(
                    np.reshape(np.roll(self.h_array, -1, axis=2), -1),
                    self.__startcoords,
                )
                - self.__start_h
            )
            ** 2
        ) * np.take(np.reshape(np.roll(mask, -1, axis=2), -1), self.__startcoords)

        st_energy += (
            (
                np.take(
                    np.reshape(np.roll(self.__update_h_array, -1, axis=2), -1),
                    self.__movecoords,
                )
                - self.__fluctmove_h
            )
            ** 2
            - (
                np.take(
                    np.reshape(np.roll(self.h_array, -1, axis=2), -1),
                    self.__movecoords,
                )
                - self.__move_h
            )
            ** 2
        ) * np.take(np.reshape(np.roll(mask, -1, axis=2), -1), self.__movecoords)

        st_energy = 0.5 * np.repeat(self.a3, self.dimension**2) * st_energy

        return gr1_energy_change + gr2_energy_change + st_energy

    def _check_update_h(self):
        """Uses the Boltzmann/Metropolis criterion to decide whether to update h for each tile.
        - Returns a Boolean 1D array of length (dim*dim)*num_trials corresponding to whether move is accepted
        - Returns a 1D array of length num_trials corresponding to the total energy change for each trial
        """

        energy_change = (
            self._calc_h_energy_change()
        )  # 1D array of length (dim*dim)*num_trials
        random_numbers = random.uniform(
            0, 1, self.dimension * self.dimension * self.num_trials
        )

        boltzmann_factor = np.exp(
            -energy_change * np.repeat(self.beta, self.dimension**2)
        )  # 1D array of length (dim*dim)*num_trials

        metropolis_factor = boltzmann_factor * (
            (np.abs(self.__start_h + self.min_fluct_h * 100 / self.fluct_h))
            / (np.abs(self.__move_h + self.min_fluct_h * 100 / self.fluct_h))
        )

        metropolis_criteria = (
            metropolis_factor > random_numbers
        )  # 1D Boolean array of length (dim*dim)*num_trials

        energy_change = (
            (energy_change * metropolis_criteria)
            .reshape(self.num_trials, -1)
            .sum(axis=1)
        )  # Total energy change for each trial

        return metropolis_criteria, energy_change

    def _check_update_phi(self):
        """Uses the Boltzmann/Metropolis criterion to decide whether to update phi for each tile.
        - Returns a Boolean 1D array of length (dim*dim)*num_trials corresponding to whether move is accepted
        - Returns a 1D array of length num_trials corresponding to the total energy change for each trial
        """

        energy_change = (
            self._calc_phi_energy_change()
        )  # 1D array of length (dim*dim)*num_trials
        random_numbers = random.uniform(
            0, 1, self.dimension * self.dimension * self.num_trials
        )

        boltzmann_factor = np.exp(
            -energy_change * np.repeat(self.beta, self.dimension**2)
        )  # 1D array of length (dim*dim)*num_trials

        metropolis_factor = boltzmann_factor * (self.__start_phi) / (self.__move_phi)

        metropolis_criteria = (
            metropolis_factor > random_numbers
        )  # 1D Boolean array of length (dim*dim)*num_trials

        energy_change = (
            (energy_change * metropolis_criteria)
            .reshape(self.num_trials, -1)
            .sum(axis=1)
        )  # Total energy change for each trial

        return metropolis_criteria, energy_change

    def make_update(self):
        """Completes a full iteration of the Metropolis algorithm:
        ------------------------------------
        - Chooses random start and move points
        - Creates phi fluctuations at these random points
        - Calculates energy change and metropolis criteria
        - Updates array if criteria are met
        - Updates energy count and array
        ------------------------------------
        - Repeats process for h fluctuations

        - Updates iteration counter
        - Every 1000 iterations updates the phi_std_array
        """

        self._choose_indices()
        self._create_phi_fluctuations()

        metropolis_criteria, energy_change = self._check_update_phi()

        # perform phi update
        np.reshape(self.phi_array, -1)[self.__startcoords] = np.where(
            metropolis_criteria,
            self.__fluctstart_phi,
            self.__start_phi,
        )
        np.reshape(self.phi_array, -1)[self.__movecoords] = np.where(
            metropolis_criteria,
            self.__fluctmove_phi,
            self.__move_phi,
        )
        self.energy_count += energy_change

        # --------------------------------------

        self._choose_indices()
        self._create_h_fluctuations()

        metropolis_criteria, energy_change = self._check_update_h()

        # perform h update
        np.reshape(self.h_array, -1)[self.__startcoords] = np.where(
            metropolis_criteria,
            self.__fluctstart_h,
            self.__start_h,
        )
        np.reshape(self.h_array, -1)[self.__movecoords] = np.where(
            metropolis_criteria,
            self.__fluctmove_h,
            self.__move_h,
        )

        self.energy_count += energy_change

        self.iteration_count += 1
        self.energy_array.append(self.energy_count)

        if self.iteration_count % 1000 == 0:
            self.phi_std = np.std(self.array[:, :, :, 0], axis=(1, 2))
            self.phi_std_array.append(self.phi_std)

    def save_lattice(self, filename):
        """Saves the current instance of the 3DLattice class using pickle
        Args:
            filename (str): Name of saved file, must end in .pkl
        """
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        """Class method used to load a previously saved class instance e.g. new_instance=3DLattice.load('filenmae.pkl')
        Args:
            filename (str): Name of saved file, must end in .pkl"""

        with open(filename, "rb") as f:
            return pickle.load(f)
