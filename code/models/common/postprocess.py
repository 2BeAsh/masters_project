import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import uniform_filter1d
import scipy.special as special
import scipy.signal
import matplotlib.pyplot as plt
import h5py
from sklearn.neighbors import KernelDensity
from run import file_path
import general_functions


class PostProcess:
    """Functions that are used to do calculations on the output of the model.

    """
    def __init__(self, data_group_name):
        self.group_name = data_group_name
        self.loaded_groups = {}  # Used to make sure does not load data multiple times
    
    
    def _load_data_group(self, gname):
        with h5py.File(file_path, "r") as f:
            group = f[gname]
            data = {
                "w": np.array(group.get("w", None)),
                "d": np.array(group.get("d", None)),
                "s": np.array(group.get("s", None)),
                "r": np.array(group.get("r", None)),
                "went_bankrupt": np.array(group.get("went_bankrupt", None)),
                "went_bankrupt_idx": np.array(group.get("went_bankrupt_idx", None)),
                "mu": np.array(group.get("mu", None)),
                "mutations": np.array(group.get("mutations", None)),
                "peak_idx": np.array(group.get("peak_idx", None)),
                "repeated_m_runs": np.array(group.get("repeated_m_runs", None)),
                "N": np.array(group.attrs.get("N", None)),
                "time_steps": group.attrs.get("time_steps"),
                "W": np.array(group.attrs.get("W", None)),
                "ds": np.array(group.attrs.get("ds", None)),
                "rf": np.array(group.attrs.get("rf", None)),
                "m": np.array(group.attrs.get("m", None)),
                "prob_expo": np.array(group.attrs.get("prob_expo", None)),
                "m_repeated": np.array(group.attrs.get("m_repeated", None)),
                "s_s_min": np.array(group.get("s_s_min", None)),
                "bankruptcy_s_min": np.array(group.get("bankruptcy_s_min", None)),
                "s_min_list": np.array(group.attrs.get("s_min_list", None)),
                "s_ds": np.array(group.get("s_ds", None)),
                "bankruptcy_ds": np.array(group.get("bankruptcy_ds", None)),
                "ds_list": np.array(group.attrs.get("ds_list", None))
            }
            self.loaded_groups[gname] = data
    
    
    def _get_data(self, gname):
        """Load data from gname if it has not already been loaded."""
        if gname not in self.loaded_groups:
            self._load_data_group(gname)
        # Set the attributes from the loaded data
        data = self.loaded_groups[gname]
        self.w = data["w"]
        self.d = data["d"]
        self.s = data["s"]
        self.r = data["r"]
        self.went_bankrupt = data["went_bankrupt"]
        self.went_bankrupt_idx = data["went_bankrupt_idx"]
        self.mu = data["mu"]
        self.mutations = data["mutations"]
        self.N = data["N"]
        self.time_steps = data["time_steps"]
        self.W = data["W"]
        self.ds = data["ds"]
        self.rf = data["rf"]
        self.m = data["m"]
        self.prob_expo = data["prob_expo"]
        self.peak_idx = data["peak_idx"]
        self.salary_repeated_m_runs = data["repeated_m_runs"]
        self.m_repeated = data["m_repeated"]
        self.s_s_min = data["s_s_min"]
        self.bankruptcy_s_min = data["bankruptcy_s_min"]
        self.s_min_list = data["s_min_list"]
        self.s_ds = data["s_ds"]
        self.bankruptcy_ds = data["bankruptcy_ds"]
        self.ds_list = data["ds_list"]
        self.time_values = np.arange(self.time_steps)
    
    
    def _skip_values(self, *variables):
        """For all array-like variables given, return the values from skip_values to the end of the array.

        Args:
            *variables: Any number of array-like variables.

        Returns:
            list: A list of sliced arrays.
        """
        size_reduced_arrays = []
        for var in variables:
            # If arr is not an array, subtract the skip_values
            if not isinstance(var, np.ndarray):
                size_reduced_arrays.append(var - self.skip_values)
            # If var is 2dim, skip values in axis=1
            elif var.ndim == 2:
                size_reduced_arrays.append(var[:, self.skip_values:])
            # var is a 1dim array
            else:
                size_reduced_arrays.append(var[self.skip_values:])
            
        return tuple(size_reduced_arrays)
        
            
    def time_from_negative_income_to_bankruptcy(self, skip_values, show_plot):
        """For each company:
            - Find all times of bankruptcy using find_peaks to avoid counting multiple bankruptcies due to bad salary choice.
            - Find the time of all changes in income from negative to positive using find_peaks
        """
        # Load data
        self._get_data(self.group_name)
        
        # Find the bottom of the debt curve using find_peaks. The distance between the bottom and the peak is the time from negative income to bankruptcy.
        
        time_diff_arr = np.zeros(self.N, dtype="object")
        
        for i in range(self.N):            
            # Get the mean and std of the time from bottom to peak
            bot_times, top_times = self._get_peak_pairs(i, skip_values, show_plot)
            time_from_bottom_to_peak = top_times - bot_times
            
            # Store values
            time_diff_arr[i] = time_from_bottom_to_peak
            
        return np.concatenate(time_diff_arr)
            
            
    def _get_peak_pairs(self, idx, skip_values, show_plot):
        d_skipped = self.d[idx, skip_values:]
        bot_height = np.abs(np.min(d_skipped) * 0.25)
        bottom_idx, _ = find_peaks(-d_skipped, height=bot_height, distance=30, prominence=bot_height, width=25)  # Maybe play around with plateau size?
        top_idx = np.where(self.went_bankrupt_idx[idx, skip_values:]==1)[0]
        
        if show_plot:
            plt.figure()
            plt.plot(np.arange(skip_values, self.time_steps), d_skipped, color="firebrick")
            
        # Check if there is any peaks
        if len(bottom_idx) == 0 or len(top_idx) == 0:
            if show_plot:
                plt.show()
                plt.close()
            return np.array([])
        
        # Need to make sure that bottom and top are equal in length
        top_idx = top_idx.tolist()
        
        # If the first peak is a top, remove it
        if top_idx[0] < bottom_idx[0]:
            top_idx = top_idx[1:]
        
        # If the last peak is a bottom, remove it
        if bottom_idx[-1] > top_idx[-1]:
            bottom_idx = bottom_idx[:-1]
        
        # Need to check again if there are any peaks after removing end points
        if len(bottom_idx) == 0 or len(top_idx) == 0:
            return np.array([])
        
        # Go through all bottom peaks and check if all greater bottom peaks are closer to it than the nearest top peak.
        # Remove all bottom peaks that satisfy this
        i_bot = 0
        finished_bot = False
        while not finished_bot:        
            bottom = bottom_idx[i_bot]
            top = top_idx[i_bot]
            
            top_bottom_diff = top - bottom
            bottom_other_bottom_diff = np.diff(bottom_idx[i_bot:])
            
            idx_to_remove = i_bot + np.where(bottom_other_bottom_diff < top_bottom_diff)[0]   
            bottom_idx = np.delete(bottom_idx, idx_to_remove)
            
            i_bot += 1
            if i_bot == len(bottom_idx):
                finished_bot = True
                
        # Because we have removed bottom duplicates, can remove top duplicates in one loop
        i_top = 0
        while i_top < len(bottom_idx) and i_top < len(top_idx):
            if top_idx[i_top] < bottom_idx[i_top]:
                top_idx.pop(i_top)
            else:
                i_top += 1
        top_idx = top_idx[:len(bottom_idx)]
        bottom_idx = bottom_idx[:len(top_idx)]

        if show_plot:
            plt.plot(np.array(bottom_idx)+skip_values, d_skipped[bottom_idx], "o", color="green", markersize=5)
            plt.plot(np.array(top_idx)+skip_values, d_skipped[top_idx], "o", color="red", markersize=5)
            # plt.axvline(x=5360, ls="dashed", color="grey")
            # plt.axvline(x=5600, ls="dashed", color="grey")
            # plt.axvline(x=5840, ls="dashed", color="grey")
            plt.xlabel("Time")
            plt.ylabel("Debt")
            plt.grid()
            plt.title("Debt first minima and bankruptcy")
            plt.show()
            plt.close()
                    
        return bottom_idx, np.array(top_idx)
                
        
    def time_diff_llh_minimize(self, skip_values, show_plot):
        diff_vals = self.time_from_negative_income_to_bankruptcy(skip_values, show_plot)

        def _double_gaussian_pdf(x, mu1, sigma1, mu2, sigma2, p):
            # Ensure only positive values
            x = x[x > 0]

            gauss1 = p * 1 / (sigma1 * np.sqrt(2 * np.pi)) * np.exp(-1/2 * (x - mu1)**2 / sigma1**2) 
            gauss2 = (1 - p) * 1 / (sigma2 * np.sqrt(2 * np.pi)) * np.exp(-1/2 * (x - mu2)**2 / sigma2**2)
            return gauss1 + gauss2
        
        def _double_log_normal_pdf(x, mu1, sigma1, mu2, sigma2, p):
            # Ensure only positive values
            x = x[x > 0]
            
            lognorm1 = p * (1 / (x * sigma1 * np.sqrt(2 * np.pi))) * np.exp(- (np.log(x) - mu1)**2 / (2 * sigma1**2))
            lognorm2 = (1 - p) * (1 / (x * sigma2 * np.sqrt(2 * np.pi))) * np.exp(- (np.log(x) - mu2)**2 / (2 * sigma2**2))
            return lognorm1 + lognorm2
        
        
        def _double_gamma_pdf(x, a1, b1, a2, b2, p):
            # Ensure only positive values
            x = x[x > 0]

            gamma1 = p * (b1**a1 / special.gamma(a1)) * x**(a1 - 1) * np.exp(-b1 * x)
            gamma2 = (1 - p) * (b2**a2 / special.gamma(a2)) * x**(a2 - 1) * np.exp(-b2 * x)
            return gamma1 + gamma2
        
        
        # Estimate initial values and perform minimization
        p0_list = [25., 14., 150, 25, 0.2]
        par, _ = general_functions.minimize_llh(_double_gaussian_pdf, diff_vals, p0=p0_list)
        
        p0_list_lognorm = [0, 1/2, 2, 1/2, 0.4]
        par_lognorm, _ = general_functions.minimize_llh(_double_log_normal_pdf, diff_vals, p0=p0_list_lognorm)
        
        p0_list_gamma = [2, 2, 8, 1, 0.4]
        par_gamma, _ = general_functions.minimize_llh(_double_gamma_pdf, diff_vals, p0=p0_list_gamma)        
        
        # Calculate x and y values for plotting
        x_values = np.linspace(np.min(diff_vals), np.max(diff_vals), 500)
        y_norm = _double_gaussian_pdf(x_values, *par)
        y_lognorm = _double_log_normal_pdf(x_values, *par_lognorm)
        y_gamma = _double_gamma_pdf(x_values, *par_gamma)
        
        # Print the fitted values for double Gaussian
        mu1, sigma1, mu2, sigma2, p = par
        print(f"Double Gaussian: mu1 = {mu1:.2f}, sigma1 = {sigma1:.2f}, mu2 = {mu2:.2f}, sigma2 = {sigma2:.2f}, p = {p:.2f}")
        
        # Print the fitted values for double log normal
        mu1_ln, sigma1_ln, mu2_ln, sigma2_ln, p_ln = par_lognorm
        print(f"Double Log Normal: mu1 = {mu1_ln:.2f}, sigma1 = {sigma1_ln:.2f}, mu2 = {mu2_ln:.2f}, sigma2 = {sigma2_ln:.2f}, p = {p_ln:.2f}")
        
        # Print the fitted values for double gamma
        a1, b1, a2, b2, p_gamma = par_gamma
        print(f"Double Gamma: a1 = {a1:.2f}, b1 = {b1:.2f}, a2 = {a2:.2f}, b2 = {b2:.2f}, p = {p_gamma:.2f}")
        
        return par, x_values, y_norm, y_lognorm, y_gamma, diff_vals
    
    
    def _survive_bust_single(self, time_interval):
        # Skip time
        went_bankrupt_idx_over_peak = self.went_bankrupt_idx[:, time_interval[0] : time_interval[1]]
        
        # Find the number of times each company goes bankrupt over that period
        company_goes_bankrupt_over_peak = np.count_nonzero(went_bankrupt_idx_over_peak==1, axis=1) 
        # Count the number of companies with 0 bankruptcies
        survive = np.count_nonzero(company_goes_bankrupt_over_peak==0)
        return survive
        
    
    def survive_bust_distribution(self, show_peak_plot):
        """The number of companies that do not go bankrupt from the bottom of one peak to the next.
        """
        # Load data
        self._get_data(self.group_name)
                
        # Smooth the mean salary data, then find the minus peaks
        s_skip = self.s[:, self.skip_values:]
        mean_s = np.mean(s_skip, axis=0)
        smooth_s = savgol_filter(mean_s, 51, 3)
        height_and_prominence = np.min(-smooth_s) / 2
        peaks, _ = find_peaks(-smooth_s, height=height_and_prominence, prominence=height_and_prominence, distance=15, width=10)
        
        if show_peak_plot:
            fig, ax = plt.subplots()
            t = np.arange(self.skip_values, self.time_steps)   
            ax.plot(t, -mean_s, label="Mean salary")
            ax.plot(t, -smooth_s, label="Smoothed mean salary")
            ax.plot(peaks+self.skip_values, -smooth_s[peaks], "x", markersize=10, label="Peaks")
            ax.set(xlabel="Time", ylabel="Salary", title="Negative mean salary smoothed peaks")
            ax.legend()
            ax.grid()
            plt.show()
            plt.close()
        
        
        # Loop over peaks to find the number of companies that survive from one peak to the next
        number_of_peaks = len(peaks)
        if number_of_peaks <= 1:
            print("No peaks found")
            return np.array([])
        else:
            survive_arr = np.zeros(number_of_peaks-1)
            for i in range(number_of_peaks-1):
                survive_arr[i] = self._survive_bust_single([peaks[i], peaks[i+1]])
            
        return survive_arr
    
    
    def single_KDE(self, x_data: str, time_point=10, bandwidth=None, eval_points=100, kernel="gaussian", s_lim=None):
        """Plot the KDE of a single time step.

        Args:
            x_data (str): Determines which data to use. Either "salary" or "debt".
            bandwidth (_type_, optional): _description_. Defaults to None.
            eval_points (int, optional): _description_. Defaults to 100.
            kernel (str, optional): _description_. Defaults to "gaussian".
            s_lim (_type_, optional): _description_. Defaults to None.
        """
        # Get data
        self._get_data(self.group_name)
        colour_name = x_data
        if x_data == "salary":
            data = self.s
        elif x_data == "debt":
            data = -self.d
        elif x_data == "delta_debt":
            data = -np.diff(self.d, axis=1)
            colour_name = "debt"
        
        s, = self._skip_values(data)
        s0 = s[:, time_point]
        # Eval points
        if np.any(s_lim == None):
            s_lim = (s0.min(), s0.max())
        s_eval = np.linspace(*s_lim, eval_points)
        # Define KDE
        if bandwidth is None:
            bandwidth = s_eval[1] - s_eval[0]
        kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)
        # Fit KDE
        kde.fit(s0[:, None])
        log_score = kde.score_samples(s_eval[:, None])
        KDE_prop = np.exp(log_score)
        
        # Histogram
        Nbins = int(np.sqrt(len(s0)))
        
        # Create figure
        fig, ax = plt.subplots()
        ax.hist(s0, bins=Nbins, label=f"{x_data} histogram", color=self.colours[colour_name], alpha=0.9, density=True)
        ax.plot(s_eval, KDE_prop, label="KDE", c="black")
        ax.set(xlabel="s", ylabel="P(s)", title=f"{kernel} KDE, bw={bandwidth:.1e}, eval_points={eval_points}")
        ax.grid()
        ax.legend()
        plt.show()
    
    
    def running_KDE(self, x_data: str, bandwidth=None, eval_points=100, kernel="gaussian", data_lim=None):
        """Loop over all time steps and calculate the KDE for each time step.
        """
        # Get data
        self._get_data(self.group_name)
        add_time = 0
        if x_data == "salary":
            x_data = self.s
        elif x_data == "debt":
            x_data = -self.d
        elif x_data == "delta_debt":
            x_data = -np.diff(self.d, axis=1)
            add_time = 1  # For delta_debt need to add an additional time step to account for the difference losing one
        data, time_steps = self._skip_values(x_data, self.time_steps-add_time)
        # Calculate points for the KDE to be evaulated at such that all times have the same evaluation points
        if np.any(data_lim == None):
            data_lim = (data.min(), data.max())
        eval = np.linspace(*data_lim, eval_points)
        # Empty array for storing KDE probabilities
        KDE_prop_arr = np.zeros((eval_points, time_steps))
        # Define KDE
        if bandwidth is None:
            bandwidth = eval[1] - eval[0]
        kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)
        # Loop over time and calculate KDE
        for i in range(time_steps):
            # Fit KDE
            s_current_time = data[:, i]
            kde.fit(s_current_time[:, None])
            log_score = kde.score_samples(eval[:, None])
            KDE_prop_arr[:, i] = np.exp(log_score)
        
        return eval, KDE_prop_arr
    
    # Power Spectral Density (PSD) analysis
    
    def _compute_PSD(self, data, fs):
        """Use Welch's method to compute the Power Spectral Density (PSD) of the data.

        Args:
            data (_type_): The data to compute the PSD of.
            fs (_type_): The sampling frequency.

        Returns:
            _type_: _description_
        """
        if len(data) == 0:
            print("No data to compute PSD, returning empty arrays")
            return np.array([]), np.array([])
        # Detrend the data
        data_detrended = scipy.signal.detrend(data)
        freqs, psd = scipy.signal.welch(data_detrended, fs=fs, nperseg=min(8192, len(data_detrended)))
        return freqs, psd

    
    def _find_dominant_frequencies(self, freqs, psd, number_of_frequencies):
        """Find the peaks in the Power spectral density (PSD) and return the number_of_frequencies most prominent peaks.

        Args:
            freqs (_type_): _description_
            psd (_type_): _description_
            number_of_frequencies (_type_): _description_

        Returns:
            _type_: _description_
        """
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
    
    
    def _PSD_on_dataset(self, data, number_of_frequencies, fs=1, show_power_plot=False):
        """Calls the _compuse_PSD and _find_dominant_frequencies to actually do the calculation.

        Args:
            data (_type_): _description_
            number_of_frequencies (_type_): _description_
            fs (_type_): _description_

        Returns:
            _type_: _description_
        """
        freqs, psd = self._compute_PSD(data, fs=fs)
        dominant_freqs, dominant_powers = self._find_dominant_frequencies(freqs, psd, number_of_frequencies)
        
        if show_power_plot:
            plt.figure()
            plt.semilogy(freqs, psd)
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("Power Spectral Density")
            plt.title("PSD of Dataset")
            plt.show()

        return dominant_freqs, dominant_powers
