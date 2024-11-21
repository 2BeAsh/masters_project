import numpy as np
from pathlib import Path
import h5py
from scipy.signal import find_peaks
from redistribution_no_m_master import dir_path_output, group_name


class PostProcessing():
    """Loads data created from master file, calculates peaks and saves the peak data to the same file
    """
    def __init__(self, data_group_name):
        # Define variable
        self.data_group_name = data_group_name
        
        self._load_data()
        
        
    def _load_data(self):
       # Check if the path to the image folder exists, otherwise create it
        self.data_path = dir_path_output / "redistribution_no_m_simulation_data.h5"
        
        # Load data        
        with h5py.File(self.data_path, "r") as file:
            data_group = file[self.data_group_name]
            # Print the names of all groups in file
            self.filename = data_group.name.split("/")[-1]
            
            # Company
            self.w = data_group["w"][:]
            self.d = data_group["d"][:]
            self.s = data_group["s"][:]
            
            # System
            self.interest_rate = data_group["interest_rate"][:]
            self.went_bankrupt = data_group["went_bankrupt"][:]
            self.unemployed = data_group["unemployed"][:]
            self.system_money_spent = data_group["system_money_spent"][:]
        
            # Attributes
            self.rho = data_group.attrs["salary_increase"]
            self.salary_increase_space = data_group.attrs["salary_increase_space"]
            self.W = data_group.attrs["W"]
            
        self.N = self.w.shape[0]
        self.time_steps = self.w.shape[1]
        self.time_values = np.arange(self.time_steps)

        
    def _peaks(self):
        # Find the amplitude and frequency of system_money_spent peaks using scipy find_peaks
        # Remove the first initial_values_skipped data points as they are warmup
        # Prominence: Take a peak and draw a horizontal line to the highest point between the peak and the next peak. The prominence is the height of the peak's summit above this horizontal line.
        initial_values_skipped = np.min((5000, self.time_steps//2))
        system_money = self.system_money_spent[initial_values_skipped:]
        prominence = system_money.max() / 20  # Prominence is 1/4 of the max value
        self.peak_idx, _ = find_peaks(x=system_money, height=5, distance=25, width=5, prominence=prominence)  # Height: Minimum y value, distance: Minimum x distance between peaks, prominence: 
        self.peak_vals = system_money[self.peak_idx]
        self.peak_idx += initial_values_skipped 


    def _get_data(self):
        # Load data, then calculate peaks
        self._load_data()
        self._peaks()
        
        
    def store_peak_data(self):
        # Store peak data in the same file as the simulation data
        # Calculate peaks
        self._get_data()
        
        with h5py.File(self.data_path, "a") as file:
            data_group = file[self.data_group_name]
            
            # Check if already has the data group, and if so, delete it
            if "peak_idx" in data_group:
                del data_group["peak_idx"]
                del data_group["peak_vals"]
                
            data_group.create_dataset("peak_idx", data=self.peak_idx)
            data_group.create_dataset("peak_vals", data=self.peak_vals)
            
            
    def store_peak_over_parameter_space(self):
        """Get peak values for different salary increase values or machine cost
        """
        # Depending on whether rho or machine cost is investigated, change the variable
        N_sim = len(self.salary_increase_space)
        for i, rho in enumerate(self.salary_increase_space):
            self.salary_increase = rho
            self.data_group_name = f"Steps{self.time_steps}_N{self.N}_W{self.W}_ds{rho}"
            self.store_peak_data()
            

if __name__ == "__main__":
    postprocess = PostProcessing(group_name)
    postprocess.store_peak_over_parameter_space()
    print("Peaks calculated and stored in file")
