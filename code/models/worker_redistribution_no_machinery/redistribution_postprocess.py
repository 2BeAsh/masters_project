import numpy as np
from pathlib import Path
import h5py
import scipy.signal
from redistribution_no_m_master import Workforce, number_of_companies, number_of_workers, salary_increase, interest_rate_free, time_steps, ds_space, rf_space, seed, time_scale_func
import joblib
from scipy.optimize import curve_fit


import matplotlib.pyplot as plt


class PostProcessing(Workforce):
    """Loads data created from master file, calculates peaks and saves the peak data to the same file
    """
    def __init__(self):
        super().__init__(number_of_companies, number_of_workers, salary_increase, interest_rate_free, time_steps, ds_space, rf_space, time_scale_func, seed)

        
    def _load_data(self):
       # Check if the path to the image folder exists, otherwise create it
        # Load data        
        self.data_group_name = self._get_group_name()

        with h5py.File(self.file_path, "r") as file:
            data_group = file[self.data_group_name]
            
            # Company
            self.s = data_group["s"][:]
            self.w = data_group["w"][:]
            
            # System
            self.went_bankrupt = data_group["went_bankrupt"][:]
            self.system_money_spent = data_group["system_money_spent"][:]
        
            # Attributes
            self.W = data_group.attrs["W"]
            self.rf_space = data_group.attrs["rf_space"]
            self.ds_space = data_group.attrs["ds_space"]
        
        self.N = self.s.shape[0]            
        self.time_steps = len(self.went_bankrupt)
        self.time_values = np.arange(self.time_steps)
    
    
    def delete_data(self):
        """Delete datagroups based on time steps"""
        with h5py.File(self.file_path, "a") as file:
            for gname in file.keys():
                if "Steps" in gname:
                    # Get the number of time steps in the group by loading an array and checking the length
                    time_steps = len(file[gname]["went_bankrupt"])
                    if time_steps < 25001:
                        del file[gname]
                        print(f"Deleted group {gname}")
                    
    
    def _compute_PSD(self, data, fs):
        if len(data) == 0:
            print("No data to compute PSD, returning empty arrays")
            return np.array([]), np.array([])
        # Detrend the data
        data_detrended = scipy.signal.detrend(data)
        freqs, psd = scipy.signal.welch(data_detrended, fs=fs, nperseg=min(8192, len(data_detrended)))
        return freqs, psd
    
    
    def _find_dominant_frequencies(self, freqs, psd, number_of_frequencies):
        if len(psd) == 0:
            print("No data to find dominant frequencies, returning empty arrays")
            return np.array([np.nan]*number_of_frequencies), np.array([np.nan]*number_of_frequencies)
        peaks, properties = scipy.signal.find_peaks(psd, prominence=np.max(psd)*0.05)
        if len(peaks) == 0:
            print("No peaks found, returning empty arrays")
            return np.array([np.nan]*number_of_frequencies), np.array([np.nan]*number_of_frequencies)
        sorted_indices = np.argsort(psd[peaks])[::-1]  # Sort in descending order
        dominant_freqs = freqs[peaks][sorted_indices][:number_of_frequencies]  # Get the the number_of_frequencies most prominent frequency peaks
        dominant_powers = psd[peaks][sorted_indices][:number_of_frequencies]
       
        # If less than number_of_frequencies peaks are found, fill the rest with NaN
        if len(dominant_freqs) < number_of_frequencies:
            dominant_freqs = np.pad(dominant_freqs, (0, number_of_frequencies-len(dominant_freqs)), constant_values=np.nan)
            dominant_powers = np.pad(dominant_powers, (0, number_of_frequencies-len(dominant_powers)), constant_values=np.nan)
        return dominant_freqs, dominant_powers
        
        
    def _PSD_on_dataset(self, data, number_of_frequencies, fs):
        freqs, psd = self._compute_PSD(data, fs=fs)
        # plt.figure()
        # plt.semilogy(freqs, psd)
        # plt.xlabel("Frequency [Hz]")
        # plt.ylabel("Power Spectral Density")
        # plt.title("PSD of Dataset")
        # plt.show()
        dominant_freqs, dominant_powers = self._find_dominant_frequencies(freqs, psd, number_of_frequencies)
        return dominant_freqs, dominant_powers
    
    
    def _PSD_parallel(self, number_of_frequencies, fs):
        # Load list of datasets to compute PSD on
        mean_salary_data_sets = []

        for ds in self.ds_space:
            self.salary_increase = ds
            for r_f in self.rf_space:
                self.interest_rate_free = r_f
                self._load_data()
                salary_without_wamup = self.s[:, 2500:]
                mean_salary = np.mean(salary_without_wamup, axis=0)
                mean_salary_data_sets.append(mean_salary)
        
        PSD_func = lambda x: self._PSD_on_dataset(x, number_of_frequencies=number_of_frequencies, fs=fs)
        
        results = joblib.Parallel(n_jobs=-1)(joblib.delayed(PSD_func)(data) for data in mean_salary_data_sets)
        
        # Reshape results to match the shape of the parameter space
        dominant_freqs_array = np.array([result[0] for result in results]).reshape((len(self.ds_space), len(self.rf_space), number_of_frequencies))
        dominant_powers_array = np.array([result[1] for result in results]).reshape((len(self.ds_space), len(self.rf_space), number_of_frequencies))    
        
        return dominant_freqs_array, dominant_powers_array
        
    
    def store_PSD_individually(self, number_of_frequencies, fs):
        dominant_freqs_array, dominant_powers_array = self._PSD_parallel(number_of_frequencies, fs)

        # Loop over the parameter space and store the PSD data
        for i, ds in enumerate(self.ds_space):
            self.salary_increase = ds
            for j, r_f in enumerate(self.rf_space):
                self.interest_rate_free = r_f
                self.data_group_name = self._get_group_name()
                
                dominant_freqs = dominant_freqs_array[i, j]
                dominant_powers = dominant_powers_array[i, j]
                
                with h5py.File(self.file_path, "a") as file:
                    data_group = file[self.data_group_name]
                    if "dominant_freqs" in data_group:
                        del data_group["dominant_freqs"]
                        del data_group["dominant_powers"]
                    data_group.create_dataset("dominant_freqs", data=dominant_freqs)
                    data_group.create_dataset("dominant_powers", data=dominant_powers)
    
    
    def store_PSD(self, number_of_frequencies, fs):
        dominant_freqs_array, dominant_powers_array = self._PSD_parallel(number_of_frequencies, fs)

        # Store the PSD data under a unique group name
        with h5py.File(self.file_path, "a") as file:
            func_name = self._func_time_scale.__name__.replace("_", "")
            PSD_gname = f"PSD_Steps{self.time_steps}_N{self.N}_W{self.W}_{func_name}"
            if PSD_gname in file:
                del file[PSD_gname]
            data_group = file.create_group(PSD_gname)
            data_group.create_dataset("ds_space", data=self.ds_space)
            data_group.create_dataset("rf_space", data=self.rf_space)
            data_group.create_dataset("dominant_freqs", data=dominant_freqs_array)
            data_group.create_dataset("dominant_powers", data=dominant_powers_array)

            print(f"Stored PSD data in {PSD_gname}")
            
            # Could delete the data used to calculate the PSD
            # Loop over parameter space
            # for ds in self.ds_space:
            #     self.salary_increase = ds
            #     for r_f in self.rf_space:
            #         self.interest_rate_free = r_f
            #         data_group_name = self._get_group_name()
            #         if data_group_name in file:
            #             del file[data_group_name]
            # print(f"Deleted {len(self.ds_space) * len(self.rf_space)} groups")

    
    def _double_sine(self, t, A1, f1, phi1, A2, f2, phi2, offset):
        return A1 * np.sin(2 * np.pi * f1 * t + phi1) + A2 * np.sin( 2 * np.pi * f2 * t + phi2) + offset
    
      
    def _fit(self, s_data, p0):
        """Find the frequency of the waves by fitting.
        """
        par, cov = curve_fit(self._double_sine, self.time_values, s_data, p0)
        std = np.sqrt(np.diag(cov))
        return par, std
    
    
    def _frequency_from_fit(self):
        """Fitting a double sine wave to determine the two frequencies of the mean salary.
        """
        # Load and preproccess data
        self._load_data()
        values_to_skip = 3000
        mean_salary = np.mean(self.s, axis=0)[values_to_skip:]
        time_values = self.time_values[values_to_skip:]
        # Fit the data
        # Initial guess
        # First wave is the high frequency low amplitude
        A1 = 1
        f1 = 1 / 530
        phi1 = 0.1
        # Second wave is the low frequency.
        A2 = 0.05
        f2 = 1 / 3400
        phi2 = 0
        
        offset = 0.1
        
        p0 = [A1, f1, phi1, A2, f2, phi2, offset]
        par, std = self._fit(np.mean(self.s, axis=0), p0)
        
        # Plot
        fig, ax = plt.subplots()
        ax.plot(time_values, mean_salary, label="Data")
        ax.plot(time_values, self._double_sine(time_values, *par), label="Fit")
        ax.set_xlabel("Time")
        ax.set(yscale="log")
        ax.grid()
        ax.legend()
        ax.text(0.3, 0.8, f"Frequency 1: {par[1]:.2e} +/- {std[1]:.2e}\nFrequency 2: {par[4]:.2e} +/- {std[4]:.2e}\n Amplitude 1: {par[0]:.2e} +- {std[0]:.2e}\n Amplitude 2: {par[3]:.2e} +/- {std[3]:.2e}", transform=ax.transAxes, fontsize=8)
        plt.show()
                
        
    def _peaks_from_system_money(self):
        # Find the amplitude and frequency of system_money_spent peaks using scipy find_peaks
        # Remove the first initial_values_skipped data points as they are warmup
        # Prominence: Take a peak and draw a horizontal line to the highest point between the peak and the next peak. The prominence is the height of the peak's summit above this horizontal line.
        initial_values_skipped = np.min((2500, self.time_steps//2))
        system_money = self.system_money_spent[initial_values_skipped:]
        prominence = system_money.max() / 20  # Prominence is 1/4 of the max value
        self.peak_idx, _ = scipy.signal.find_peaks(x=system_money, height=5, distance=25, width=5, prominence=prominence)  # Height: Minimum y value, distance: Minimum x distance between peaks, prominence: 
        self.peak_vals = system_money[self.peak_idx]
        self.peak_idx += initial_values_skipped 


    def _peaks_from_bankruptcy(self):
        # Get peaks from went_bankrupt
        initial_values_skipped = np.min((1000, self.time_steps//2))
        bankrupt = self.went_bankrupt / self.N
        self.peak_idx, _ = scipy.signal.find_peaks(x=bankrupt[initial_values_skipped:], height=0.05, prominence=0.05, distance=50,)
        self.peak_idx += initial_values_skipped
        self.peak_vals = bankrupt[self.peak_idx]


    def _peaks_from_mean_salary(self):
        # Get peaks from mean salary
        initial_values_skipped = np.min((1000, self.time_steps//2))
        mean_salary = np.mean(self.s, axis=0)
        self.peak_idx, _ = scipy.signal.find_peaks(x=mean_salary[initial_values_skipped:], height=2.5e-2, prominence=2e-2, distance=100, width=100)
        self.peak_idx += initial_values_skipped
        self.peak_vals = mean_salary[self.peak_idx]


    def _get_period(self):
        """Calculate period of the peaks, exluding the first peak as it is usually in the warmup phase"""
        self.period = np.diff(self.peak_idx[1:])


    def percent_responsible_for_mu(self):
        """Find the percent of companies responsible for the majority of the production.
        """
        self._load_data()
        # Remove warmup phase
        warmup_steps = 2500
        s = self.s[:, warmup_steps:]
        w = self.w[:, warmup_steps:]
        time_values = self.time_values[warmup_steps:]
        
        # Calculate the salary paid by each company        
        salary_paid = s * w
        total_salary_paid = np.sum(salary_paid, axis=0)
        salary_paid_norm = salary_paid / total_salary_paid
        
        percent_responsible = []
        
        for i in range(len(time_values)):
            # Calculate the order of most salary paid
            order = np.argsort(salary_paid_norm[:, i])[::-1]  # Big to small
            # print(order[:5])
            # Calculate the cumulative sum
            cum_sum = np.cumsum(salary_paid_norm[order, i])
            # print(cum_sum[:5])
            # Find the index where the sum is greater than 0.5
            idx = np.where(cum_sum > 0.5)[0][0]
            percent_responsible.append(idx / self.N)
        
        # Plot
        fig, ax = plt.subplots()
        ax.plot(time_values, percent_responsible)
        ax.set(xlabel="Time", ylabel=r"Percent of companies responsible for 50% of the production")
        ax.grid()
        plt.show()
        
        
        # Store values
        with h5py.File(self.file_path, "a") as file:
            data_group = file[self.data_group_name]
            if "percent_responsible" in data_group:
                del data_group["percent_responsible"]
            data_group.create_dataset("percent_responsible", data=percent_responsible)        



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

    # postprocess.delete_data()
    

    # postprocess.store_peak_data()
    # postprocess.store_peak_over_parameter_space()
    # postprocess.store_peak_over_ds_rf_space()
    postprocess.store_PSD(number_of_frequencies=2, fs=1)

    
    # postprocess.percent_responsible_for_mu()
        
    print("Finished postprocessing")