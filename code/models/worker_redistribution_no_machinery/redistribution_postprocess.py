import numpy as np
from pathlib import Path
import h5py
from scipy.signal import find_peaks
from redistribution_no_m_master import Workforce, number_of_companies, number_of_workers, salary_increase, interest_rate_free, time_steps, ds_space, rf_space, seed


class PostProcessing(Workforce):
    """Loads data created from master file, calculates peaks and saves the peak data to the same file
    """
    def __init__(self):
        super().__init__(number_of_companies, number_of_workers, salary_increase, interest_rate_free, time_steps, ds_space, rf_space, seed)

        
    def _load_data(self):
       # Check if the path to the image folder exists, otherwise create it
        # Load data        
        with h5py.File(self.file_path, "r") as file:
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
            self.system_money_spent = data_group["system_money_spent"][:]
        
            # Attributes
            self.rho = data_group.attrs["salary_increase"]
            self.W = data_group.attrs["W"]
            self.rf_space = data_group.attrs["rf_space"]
            self.ds_space = data_group.attrs["ds_space"]
            
        self.N = self.w.shape[0]
        self.time_steps = self.w.shape[1]
        self.time_values = np.arange(self.time_steps)
    
        
    def _peaks_from_system_money(self):
        # Find the amplitude and frequency of system_money_spent peaks using scipy find_peaks
        # Remove the first initial_values_skipped data points as they are warmup
        # Prominence: Take a peak and draw a horizontal line to the highest point between the peak and the next peak. The prominence is the height of the peak's summit above this horizontal line.
        initial_values_skipped = np.min((2500, self.time_steps//2))
        system_money = self.system_money_spent[initial_values_skipped:]
        prominence = system_money.max() / 20  # Prominence is 1/4 of the max value
        self.peak_idx, _ = find_peaks(x=system_money, height=5, distance=25, width=5, prominence=prominence)  # Height: Minimum y value, distance: Minimum x distance between peaks, prominence: 
        self.peak_vals = system_money[self.peak_idx]
        self.peak_idx += initial_values_skipped 


    def _peaks_from_bankruptcy(self):
        # Get peaks from went_bankrupt
        initial_values_skipped = np.min((1000, self.time_steps//2))
        bankrupt = self.went_bankrupt / self.N
        self.peak_idx, _ = find_peaks(x=bankrupt[initial_values_skipped:], height=0.05, prominence=0.05, distance=50,)
        self.peak_idx += initial_values_skipped
        self.peak_vals = bankrupt[self.peak_idx]


    def _peaks_from_mean_salary(self):
        # Get peaks from mean salary
        initial_values_skipped = np.min((1000, self.time_steps//2))
        mean_salary = np.mean(self.s, axis=0)
        self.peak_idx, _ = find_peaks(x=mean_salary[initial_values_skipped:], height=2.5e-2, prominence=2e-2, distance=100, width=100)
        self.peak_idx += initial_values_skipped
        self.peak_vals = mean_salary[self.peak_idx]


    def _get_period(self):
        """Calculate period of the peaks, exluding the first peak as it is usually in the warmup phase"""
        self.period = np.diff(self.peak_idx[1:])


    def _get_data(self, func_peak):
        """_summary_

        Args:
            func_peak (function): Must define self.peak_vals and self.peak_idx
        """
        # Load data, then calculate peaks
        self._load_data()
        func_peak()
        self._get_period()
        
        
    def store_peak_data(self):
        # Store peak data in the same file as the simulation data
        # Calculate peaks
        self.data_group_name = self._get_group_name()
        self._get_data(func_peak=self._peaks_from_mean_salary)
        
        with h5py.File(self.file_path, "a") as file:
            data_group = file[self.data_group_name]
            
            # Check if already has the data group, and if so, delete it
            if "peak_idx" in data_group:
                del data_group["peak_idx"]
                del data_group["peak_vals"]
            if "peak_period" in data_group:          
                del data_group["peak_period"]

            data_group.create_dataset("peak_idx", data=self.peak_idx)
            data_group.create_dataset("peak_vals", data=self.peak_vals)
            data_group.create_dataset("peak_period", data=self.period)
            
            
    def store_peak_over_parameter_space(self):
        """Get peak values for different salary increase values or machine cost
        """
        # Depending on whether rho or machine cost is investigated, change the variable
        for rho in self.ds_space:
            self.salary_increase = rho
            self.store_peak_data()
    
    
    def store_peak_over_ds_rf_space(self):
        """Peak value for different ds and rf values.
        """
        for ds in self.ds_space:
            self.salary_increase = ds
            for r_f in self.rf_space:
                # Get data
                self.interest_rate_free = r_f
                self.store_peak_data()
                

if __name__ == "__main__":
    postprocess = PostProcessing()
    # postprocess.store_peak_data()
    # postprocess.store_peak_over_parameter_space()
    postprocess.store_peak_over_ds_rf_space()
    print("Peaks calculated and stored in file")
