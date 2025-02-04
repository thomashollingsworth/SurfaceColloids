"""Tweaking the original Lattice class to work with multiple trials simulataneously by using a 3D lattice array
- This will hopefully be transferrable to CuPy and GPU acceleration"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.random as random

# Used for saving data
import csv
import datetime
import pickle

class 3DLattice:
    def __init__(
        self,
        num_trials: int,
        
    ) -> None:

    
        #Initialise parameters for each trial with default values

        self.num_trials=num_trials

        self.a1=np.ones(num_trials)*12
        self.a2=np.ones(num_trials)*13
        self.a3=np.ones(num_trials)*2000
        self.a4=np.ones(num_trials)*1.3
        self.a5=np.ones(num_trials)*20000

        self.beta=np.ones(num_trials)*20000

        self.initial_phi = np.ones(num_trials)*0.001 #Mean concentration (colloids per lattice point)
    
        #These will most likely be consistent across all trials

        self.fluct_phi=5,  # Max. percentage for phi fluctuations at each step (+- fluct_phi%)
        self.fluct_h=5,  # Max. percentage change of h
        self.min_fluct_h = 0.0001 # Min. size for a h fluctuation
        self.dimension=25 #Num of 4x4 grids per row/column
    
        #Initialise array to store data
        """This array has size (dimension x dimension x num_trials x 2) where the factor of 2 corresponds to phi,h values at each point respectively
        - h values are set uniformly to 0
        - phi values are set uniformly to initial_phi for each trial
        """
        self.array=np.zeros((self.num_trials,4*(self.dimension+1),4*(self.dimension+1), 2))
        self.array[:,:,:,0] = self.initial_phi[:, np.newaxis, np.newaxis]

        self.update_array=self.array.copy() #Array to store updated values before they are accepted
    
        #Initialise boolean masks which will be used to select relevant indices in each update

        self.__startpoints=np.zeros((num_trials, 4(*self.dimension+1), 4*(self.dimension+1)))
        self.__movepoints=np.zeros((num_trials, 4*(self.dimension+1), 4*(self.dimension+1)))

        #Arrays to store energy of each trial
        self.energy_count=np.zeros(num_trials)
        self.energy_array=[]
        self.energy_change
    
        #Arrays to store iterations and other properties of interest
        self.iteration_count=0
        self.phi_std,self.phi_mean = np.zeros(num_trials), np.zeros(num_trials)
        self.h_mean, self.h_std = np.zeros(num_trials), np.zeros(num_trials)

        self.phi_std_array=[]


    def calc_absolute_energy(self):
        """Calculates the energy of each trial with a 0 point corresponding to uniform phi=initial_phi and h=0
        - This method is more costly than just calculating the energy change at each step so must be called explicitly
    
        Returns: 1D np.array of length num_trials containing the 'absolute' energy of each trial"""
    
        gr1_energy= self.a2*np.sum(self.array[:,:,:,0]*self.array[:,:,:,1],axis=(1,2))
        gr2_energy= self.a4*np.sum(self.array[:,:,:,1]**2,axis=(1,2))
    
        #Calc. electrostatic energy relative to a uniform phi array
        uniform_phi=np.mean(self.array[:,:,:,0],axis=(1,2))[:,np.newaxis,np.newaxis]*np.ones((1,4*(self.dimension+1),4*(self.dimension+1)))

        el_energy= self.a1*np.sum((self.array[:,:,:,0]**(5/2)-uniform_phi**5/2),axis=(1,2))
    
        # This is non-trivial, involves considering an alternating diagonal splitting triangular mesh and calculating the area change of each triangle
        # Area calculations use a first order approx in fluct_h/L where L is the real space distance between lattice points

        mask=np.pad(np.ones((4*(self.dimension+1),4*(self.dimension+1))), pad_width=1, mode="constant", constant_values=0
        )
        # Duplicate the mask to have num_trials copies
        mask_3D = np.repeat(mask[np.newaxis,:, :], self.num_trials, axis=0)

        # Create a checkerboard pattern of indices
        checkerboard = np.indices((4*(self.dimension+1),4*(self.dimension+1))).sum(axis=0) % 2 == 0
        checkerboard_3D=np.repeat(checkerboard[np.newaxis, :, :], self.num_trials, axis=0)
        padded_checkerboard = np.pad(
            checkerboard, pad_width=1, mode="constant", constant_values=False
        )
        padded_checkerboard_3D = np.repeat(padded_checkerboard[np.newaxis,:, :], self.num_trials, axis=0)

        h0_array=self.array[:,:,:,1][checkerboard_3D] #This is a 1D array ordered as [all checkerboard points in trial i],[all checkerboard points in trial i+1],...

        st_energy = ((np.roll(self.array[:,:,:,1],1,axis=1)[checkerboard_3D]-h0_array)* np.roll(mask_3D, 1, axis=1)[padded_checkerboard_3D])**2
        st_energy += ((np.roll(self.array[:,:,:,1],-1,axis=1)[checkerboard_3D]-h0_array)* np.roll(mask_3D, -1, axis=1)[padded_checkerboard_3D])**2
        st_energy += ((np.roll(self.array[:,:,:,1],1,axis=2)[checkerboard_3D]-h0_array)* np.roll(mask_3D, 1, axis=2)[padded_checkerboard_3D])**2
        st_energy += ((np.roll(self.array[:,:,:,1],-1,axis=2)[checkerboard_3D]-h0_array)* np.roll(mask_3D, -1, axis=2)[padded_checkerboard_3D])**2

        #st_energy is a 1D array of length (checkerboard points in one trial)*num_trials
        #need to condense it to a 1D array of length num_trials

         st_energy=st_energy.reshape(self.num_trials,-1).sum(axis=1)


        st_energy=0.5*self.a3*np.sum(st_energy,axis=(1,2))

        return gr1_energy+gr2_energy+el_energy+st_energy

    def set_initial_conditions(self):
        """- (Re)Sets initial energy and iteration counter
        - (Re)Sets energy and phi std arrays
        Should call this explicitly before running a test
        """
        self.energy_count=self.calc_absolute_energy()
        self.phi_std_array=[]
        self.iteration_count=0
        self.energy_array=[]

    def draw_fields(self,trial_num, save=False, name=None):
        """Plots the phi and h fields for a given trial
        Gives option to save the plots with a name"""

        figure = plt.figure()
        axes_h = figure.add_subplot(121)
        axes_phi = figure.add_subplot(122)

        h_plot = axes_h.matshow(self.array[trial_num,:,:,1], cmap="plasma")
        phi_plot = axes_phi.matshow(self.array[trial_num,:,:,0], cmap="plasma")

        axes_h.set_title("Height")
        axes_phi.set_title("Concentration")
        axes_h.set_axis_off()
        axes_phi.set_axis_off()

        figure.colorbar(h_plot)
        figure.colorbar(phi_plot)

        plt.tight_layout()

        if save:
            plt.savefig(f"{name}.png",dpi=300)
    
        plt.show()

    def update_all_stats(self):
        """Updates the height and phi mean and standard deviation attributes.
        [To save computation time, this method is not included in every call of make_update()]
        """

        self.h_mean, self.h_std = np.mean(self.array[:,:,:,1],axis=(1,2)), np.std(self.array[:,:,:,1],axis=(1,2))
        self.phi_mean, self.phi_std = np.mean(self.array[:,:,:,0],axis=(1,2)), np.std(self.array[:,:,:,0],axis=(1,2))

    def _choose_indices(self):
        """- Chooses random start point within 4x4 tile (same for each tile in the same 2D lattice but differs for different trials)
        - Chooses random move point within 4x4 tile (different for every tile and trial)
        - Stores results in the form of Boolean masks: self.__start/movepoints """

        start_tile=np.zeros((self.num_trials,4,4))
        start_indices=np.indices(start_tile.shape)[:,:,0,0]

        #Random generation
        random_startpoints=np.random.randint(0,4,(2,self.num_trials))

        start_indices[1:,:]=random_startpoints
        start_tile[tuple(start_indices)]=1
        
        #Mask is made up of 4x4 tiles
        start_mask=np.tile(start_tile,(self.dimension,self.dimension))
        pad_width = ((0, 0), (2, 2), (2, 2))
        start_mask=np.pad(start_mask,pad_width=pad_width,mode="constant",constant_values=0).astype(bool)
        
        self.__startpoints=start_mask #A boolean mask for all the initial points

        all_start_indices=np.indices(start_mask.shape)[:,start_mask]
        move_choices = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (-1, 1), (1, -1)]
        num_moves = all_start_indices.shape[1]

        moves = np.array(move_choices)[np.random.choice(len(move_choices), num_moves)]
        moves = moves.T

        all_move_indices = all_start_indices.copy() #Initialising
        all_move_indices[1:] += moves #Adding move data

        move_mask = np.zeros_like(start_mask,dtype=bool) #Initialising
        move_mask[tuple(all_move_indices)] = 1 #Adding move data

        self.__movepoints=move_mask #A boolean mask for all the move points

    def _create_phi_fluctuations(self):
        """Creates random phi fluctuations and then applies them to the relevant lattice points:

        - A series of random fluctuations are made [-1,1] and scaled by self.fluct_phi %
        - Negative fluctations are scaled by phi at start position
        - Positive fluctuations are scaled by phi at end position
        - Fluctuations are added and subtracted to relevant lattice points on the self.update_array
        """
        self.update_array=self.array.copy()
        #1D arrays of all relevant phi values
        start_phi=self.array[self.__startpoints,0] #1D arrays of length (start points per trial)*num_trials
        move_phi=self.array[self.__movepoints,0]
        num_moves=len(start_phi)
        fluctuations=random.uniform(-1,1,num_moves)*self.fluct_phi/100

        scaled_fluctuations = np.where(
            (fluctuations >= 0), fluctuations * movephi, fluctuations * startphi
        )
        startphi = startphi + scaled_fluctuations
        movephi = movephi - scaled_fluctuations
        
        #Adding fluctuations to the update array
        self.update_array[self.__startpoints,0]=start_phi
        self.update_array[self.__movepoints,0]=move_phi
    
    def _create_h_fluctuations(self):
        """Creates random h fluctuations and then applies them to the relevant lattice points:

        - h fluctations are: U*fluct_h/100*(|h|+(min_fluct_h*100/fluct_h)) (U is from random uniform -1:1)
        - Negative fluctations are scaled by h at start position
        - Positive fluctuations are scaled by h at end position
        - Fluctuations are added and subtracted to relevant lattice points on the self.update_array
        """

        self.update_array=self.array.copy()
        #1D arrays of all relevant h values
        start_h=self.array[self.__startpoints,1] 
        move_h=self.array[self.__movepoints,1]
        num_moves=len(start_h)
        fluctuations = random.uniform(-1, 1, num_moves) * self.fluct_h / 100
        fluct_constant = self.min_fluct_h * (100 / self.fluct_h)

        scaled_fluctuations = np.where(
            (fluctuations >= 0),
            fluctuations * (np.abs(moveh) + fluct_constant),
            fluctuations * (np.abs(starth) + fluct_constant),
        )
        starth = starth + scaled_fluctuations
        moveh = moveh - scaled_fluctuations

        #Adding fluctuations to the update array
        self.update_array[self.__startpoints,1]=start_phi
        self.update_array[self.__movepoints,1]=move_phi
    
    def _calc_phi_energy_change(self)->np.ndarray:
        """Calculates the energy change for a phi update. 
        Will return an energy change associated with each tile for each trial i.e. a 1D array of size (dimension*dimension)*num_trials
        """
        #Calculating colloid gravitational energy change
        gr1_energy_change = np.repeat(self.a2,self.dimension**2)*(self.update_array[self.__startpoints,0]*self.update_array[self.__startpoints,1]-self.array[self.__startpoints,0]*self.array[self.__startpoints,1])
        gr1_energy_change +=np.repeat(self.a2,self.dimension**2)*(self.update_array[self.__movepoints,0]*self.update_array[self.__movepoints,1]-self.array[self.__movepoints,0]*self.array[self.__movepoints,1])
        
        #Calculating electrostatic energy change
        el_energy_change= np.repeat(self.a1,self.dimension**2)*(self.update_array[self.__startpoints,0]**(5/2)-self.array[self.__startpoints,0]**(5/2))
        el_energy_change += np.repeat(self.a1,self.dimension**2)*(self.update_array[self.__movepoints,0]**(5/2)-self.array[self.__movepoints,0]**(5/2))

        return gr1_energy_change+el_energy_change
    
    def _calc_h_energy_change(self):
        """Calculates the energy change for an h update. 
        Will return an energy change associated with each tile for each trial i.e. a 1D array of size (dimension*dimension)*num_trials
        """
        #Calculating colloid gravitational energy change
        gr1_energy_change = np.repeat(self.a2,self.dimension**2)*(self.update_array[self.__startpoints,0]*self.update_array[self.__startpoints,1]-self.array[self.__startpoints,0]*self.array[self.__startpoints,1])
        gr1_energy_change +=np.repeat(self.a2,self.dimension**2)*(self.update_array[self.__movepoints,0]*self.update_array[self.__movepoints,1]-self.array[self.__movepoints,0]*self.array[self.__movepoints,1])

        #Calculating gravitational energy change of fluid
        gr2_energy_change = np.repeat(self.a4,self.dimension**2)*(self.update_array[self.__startpoints,1]**2-self.array[self.__startpoints,1]**2)
        gr2_energy_change += np.repeat(self.a4,self.dimension**2)*(self.update_array[self.__movepoints,1]**2-self.array[self.__movepoints,1]**2)

        #Calculating surface tension energy change
        mask= ~self.__movepoints
        #These are all 1D arrays of length (start points per trial)*num_trials
        self.update_array[self.__startpoints,1]= new_h_start
        self.update_array[self.__movepoints,1]= new_h_move

        self.array[self.__startpoints,1]= old_h_start
        self.array[self.__movepoints,1]= old_h_move

        st_energy = ((np.roll(self.update_array[:,:,:,1],1,axis=1)[self.__startpoints]-new_h_start)**2-(np.roll(self.array[:,:,:,1],1,axis=1)[self.__startpoints]-old_h_start)**2)*np.roll(mask,1,axis=1)[self.__startpoints]
        st_energy +=((np.roll(self.update_array[:,:,:,1],1,axis=1)[self.__movepoints]-new_h_move)**2-(np.roll(self.array[:,:,:,1],1,axis=1)[self.__movepoints]-old_h_move)**2)

        st_energy += ((np.roll(self.update_array[:,:,:,1],-1,axis=1)[self.__startpoints]-new_h_start)**2-(np.roll(self.array[:,:,:,1],-1,axis=1)[self.__startpoints]-old_h_start)**2)*np.roll(mask,-1,axis=1)[self.__startpoints]
        st_energy +=((np.roll(self.update_array[:,:,:,1],-1,axis=1)[self.__movepoints]-new_h_move)**2-(np.roll(self.array[:,:,:,1],-1,axis=1)[self.__movepoints]-old_h_move)**2)

        st_energy += ((np.roll(self.update_array[:,:,:,1],1,axis=2)[self.__startpoints]-new_h_start)**2-(np.roll(self.array[:,:,:,1],1,axis=2)[self.__startpoints]-old_h_start)**2)*np.roll(mask,1,axis=2)[self.__startpoints]
        st_energy +=((np.roll(self.update_array[:,:,:,1],1,axis=2)[self.__movepoints]-new_h_move)**2-(np.roll(self.array[:,:,:,1],1,axis=2)[self.__movepoints]-old_h_move)**2)

        st_energy += ((np.roll(self.update_array[:,:,:,1],-1,axis=2)[self.__startpoints]-new_h_start)**2-(np.roll(self.array[:,:,:,1],-1,axis=2)[self.__startpoints]-old_h_start)**2)*np.roll(mask,-1,axis=2)[self.__startpoints]
        st_energy +=((np.roll(self.update_array[:,:,:,1],-1,axis=2)[self.__movepoints]-new_h_move)**2-(np.roll(self.array[:,:,:,1],-1,axis=2)[self.__movepoints]-old_h_move)**2)
        
        st_energy=0.5*np.repeat(self.a3,self.dimension**2)*st_energy
        
        return gr1_energy_change+gr2_energy_change+st_energy

     def _check_update_h(self)->numpy.ndarray,numpy.ndarray:
        """Uses the Boltzmann/Metropolis criterion to decide whether to update h for each tile.
        - Returns a Boolean 1D array of length (dim*dim)*num_trials corresponding to whether move is accepted
        - Returns a 1D array of length num_trials corresponding to the total energy change for each trial
           """

        energy_change=self._calc_h_energy_change()#1D array of length (dim*dim)*num_trials
        random_numbers=random.uniform(0,1,self.dimension*self.dimension*self.num_trials)
    
        boltzmann_factor=np.exp(-energy_change*np.repeat(self.beta,self.dimension**2))#1D array of length (dim*dim)*num_trials

        metropolis_factor=boltzmann_factor*((np.abs(self.array[self.__startpoints,1]+self.min_fluct_h*100/self.fluct_h))/(np.abs(self.array[self.__movepoints,1]+self.min_fluct_h*100/self.fluct_h)))
        

        metropolis_criteria=metropolis_factor>random_numbers #1D Boolean array of length (dim*dim)*num_trials

        energy_change=(energy_change*metropolis_criteria).reshape(self.num_trials,-1).sum(axis=1) #Total energy change for each trial
        
        return metropolis_criteria, energy_change
    
    def _check_update_phi(self)->numpy.ndarray,numpy.ndarray:
        """Uses the Boltzmann/Metropolis criterion to decide whether to update phi for each tile.
        - Returns a Boolean 1D array of length (dim*dim)*num_trials corresponding to whether move is accepted
        - Returns a 1D array of length num_trials corresponding to the total energy change for each trial
           """
        
        energy_change=self._calc_phi_energy_change()#1D array of length (dim*dim)*num_trials
        random_numbers=random.uniform(0,1,self.dimension*self.dimension*self.num_trials)
    
        boltzmann_factor=np.exp(-energy_change*np.repeat(self.beta,self.dimension**2))#1D array of length (dim*dim)*num_trials

        metropolis_factor=boltzmann_factor*(self.array[self.__startpoints,0])/(self.array[self.__movepoints,0])
        

        metropolis_criteria=metropolis_factor>random_numbers #1D Boolean array of length (dim*dim)*num_trials

        energy_change=(energy_change*metropolis_criteria).reshape(self.num_trials,-1).sum(axis=1) #Total energy change for each trial
        
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
        
        metropolis_criteria, energy_change=self._check_update_phi()

        #perform phi update
        self.array[self.__startpoints,0]= np.where(metropolis_criteria,self.update_array[self.__startpoints,0],self.array[self.__startpoints,0])
        self.array[self.__movepoints,0]= np.where(metropolis_criteria,self.update_array[self.__movepoints,0],self.array[self.__movepoints,0])
        self.energy_count+=energy_change

        #--------------------------------------

        self._choose_indices()
        self._create_h_fluctuations()
        
        metropolis_criteria, energy_change=self._check_update_h()

        #perform h update
        self.array[self.__startpoints,1]= np.where(metropolis_criteria,self.update_array[self.__startpoints,1],self.array[self.__startpoints,1])
        self.array[self.__movepoints,1]= np.where(metropolis_criteria,self.update_array[self.__movepoints,1],self.array[self.__movepoints,1])
        
        self.energy_count+=energy_change

        self.iteration_count+=1
        self.energy_array.append(self.energy_count)

        if self.iteration_count%1000==0:
            self.phi_std= np.std(self.array[:,:,:,0],axis=(1,2))
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


        

      

        
        
        
        



        

    








