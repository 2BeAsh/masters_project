import numpy as np
import pandas as pd
import pandas_datareader.data as web
from scipy.signal import find_peaks, savgol_filter
import scipy.special
from scipy.ndimage import uniform_filter1d
import scipy.signal
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.ticker as ticker  
import h5py
import re
from sklearn.neighbors import KernelDensity
from run import dir_path_data, file_path, dir_path_output
import general_functions
import yfinance as yf
import time


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
                "w_paid": np.array(group.get("w_paid"), None),
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
                "s_min": np.array(group.attrs.get("s_min", None)),
                "s_s_min": np.array(group.get("s_s_min", None)),
                "bankruptcy_s_min": np.array(group.get("bankruptcy_s_min", None)),
                "s_min_list": np.array(group.attrs.get("s_min_list", None)),
                "s_ds": np.array(group.get("s_ds", None)),
                "bankruptcy_ds": np.array(group.get("bankruptcy_ds", None)),
                "ds_list": np.array(group.attrs.get("ds_list", None)),
                "seed": np.array(group.attrs.get("seed", None)),
            }
            self.loaded_groups[gname] = data
    
    
    def _get_data(self, gname):
        """Load data from gname if it has not already been loaded."""
        if gname is None: gname = self.group_name
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
        self.w_paid = data["w_paid"]
        self.N = data["N"]
        self.time_steps = data["time_steps"]
        self.W = data["W"]
        self.ds = data["ds"]
        self.rf = data["rf"]
        self.m = data["m"]
        self.s_min = data["s_min"]
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
        self.seed = data["seed"]
        if self.time_steps is not None:
            self.time_values = np.arange(self.time_steps)
        else: 
            self.time_values = None


    def new_gname(
        self,
        time_steps=None,
        N=None,
        W=None,
        ds=None,
        mutation_magnitude=None,
        rf_name=None,
        prob_exponent=None,
        salary_min=None,
        seed=None,
        who_want_to_increase=None,
        number_of_transactions_per_step=1,
        inject_money_time=0,
    ):
        time_steps = time_steps if time_steps is not None else self.time_steps
        N = N if N is not None else self.N
        W = W if W is not None else self.W
        ds = ds if ds is not None else self.ds
        mutation_magnitude = mutation_magnitude if mutation_magnitude is not None else self.m
        rf_name = 0.
        prob_exponent = prob_exponent if prob_exponent is not None else self.prob_expo
        salary_min = salary_min if salary_min is not None else self.s_min
        seed = seed if seed is not None else self.seed
        who_want_to_increase = who_want_to_increase if who_want_to_increase is not None else "w0"

        return (
            f"Steps{time_steps}_N{N}_W{W}_ds{ds}_m{mutation_magnitude}_rf{rf_name}"
            f"_alpha{prob_exponent}_smin{salary_min}_seed{seed}_increase{who_want_to_increase}"
            f"_transactionsfactor{number_of_transactions_per_step}_injectmoney{inject_money_time}"
        )

    
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
        
        if len(size_reduced_arrays) == 1:
            return size_reduced_arrays[0]
        
        return tuple(size_reduced_arrays)


    def _get_par_from_name(self, gname: str):
        # Get alpha, N and W from gname
        # Get the value of alpha by taking the first letter of the string just after alpha, as alpha is always a single digit
        alpha_value = gname.split("alpha")[1][0]
        # N and W maybe be multiple digits, so we need to split the string by "N", take the first element, then split by "_" and take the first element
        N_value = gname.split("N")[1].split("_")[0]
        W_value = gname.split("W")[1].split("_")[0]
        
        # Combine all values in a dictionary
        par_dict = {"alpha": alpha_value, "N": N_value, "W": W_value}
        return par_dict
    
    
    def _axis_labels_outer(self, axis, x_label, y_label, remove_y_ticks=True, remove_x_ticks=True):
        subplot_spec = axis.get_subplotspec()
        if subplot_spec.is_first_col():
            axis.set_ylabel(y_label)
        elif remove_y_ticks:
            axis.set_yticklabels([]) 
        if subplot_spec.is_last_row():
            axis.set_xlabel(x_label)
        elif remove_x_ticks:
            axis.set_xticklabels([])


    def _axis_ticks_and_labels(self, axis,
                            x_ticks=None, y_ticks=None,
                            x_labels=None, y_labels=None,
                            x_dtype=None, y_dtype=None,
                            tick_width=1.6, tick_width_minor=0.8,
                            tick_length=6, tick_length_minor=3):
        """Set major and minor ticks and labels for both axes.
        
        - x_labels/y_labels can be a list of tick values (auto labels)
        or a dict mapping tick values to custom label strings.
        """
        # Handle x-axis
        if np.all(x_ticks != None) and np.all(x_labels != None):
            major_x, major_x_labels, minor_x = [], [], []
            for tick in x_ticks:
                if isinstance(x_labels, dict):
                    if tick in x_labels:
                        major_x.append(tick)
                        major_x_labels.append(x_labels[tick])
                    else:
                        minor_x.append(tick)
                elif tick in x_labels:
                    major_x.append(tick)
                    if x_dtype == "int":
                        major_x_labels.append(f"{int(tick)}")
                    else:
                        major_x_labels.append(f"{tick}")
                else:
                    minor_x.append(tick)
            # Set ticks and labels
            axis.set_xticks(major_x)
            axis.set_xticklabels(major_x_labels)
            axis.set_xticks(minor_x, minor=True)

        # Handle y-axis
        if np.all(y_ticks != None) and np.all(y_labels != None):
            major_y, major_y_labels, minor_y = [], [], []
            for tick in y_ticks:
                if isinstance(y_labels, dict):
                    if tick in y_labels:
                        major_y.append(tick)
                        major_y_labels.append(y_labels[tick])
                    else:
                        minor_y.append(tick)
                elif tick in y_labels:
                    major_y.append(tick)
                    if y_dtype == "int":
                        major_y_labels.append(f"{int(tick)}")
                    else:
                        major_y_labels.append(f"{tick}")
                else:
                    minor_y.append(tick)
            # Set ticks and labels
            axis.set_yticks(major_y)
            axis.set_yticklabels(major_y_labels)
            axis.set_yticks(minor_y, minor=True)

        axis.tick_params(axis="both", width=tick_width, length=tick_length, which="major")
        axis.tick_params(axis="both", width=tick_width_minor, length=tick_length_minor, which="minor")

    
    def _axis_log_ticks_and_labels(self, axis, exponent_range, labels_skipped=1, numticks=10, base=10.0, which="y"):
        """_summary_

        Args:
            axis (_type_): _description_
            exponent_range (tuple of ints): The range of exponents to show on the axis.
            labels_skipped (int, optional): Every ticklabel will have labels_skipped major ticks with no labels. Defaults to 1.
            numticks (int, optional): Number of minor ticks between major ticks. Defaults to 10.
            base (float, optional): Logbase. Defaults to 10.0.
        """
        # Make sure the exponent range has integer values
        exponent_range = (np.floor(exponent_range[0]).astype(int), 
                          np.ceil(exponent_range[1]).astype(int))
        
        if which == "y":
            axis.yaxis.set_major_locator(ticker.LogLocator(base=base, numticks=numticks))
            axis.yaxis.set_minor_locator(ticker.LogLocator(base=base, subs='auto', numticks=numticks))
            # axis.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:.0e}'.format(y) if y in [base**i for i in range(*exponent_range, labels_skipped+1)] else ''))
            axis.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: r'${{10^{{{}}}}}$'.format(int(np.log10(y))) if y in [base**i for i in range(*exponent_range, labels_skipped+1)] else ''))

        elif which == "x":
            axis.xaxis.set_major_locator(ticker.LogLocator(base=base, numticks=numticks))
            axis.xaxis.set_minor_locator(ticker.LogLocator(base=base, subs='auto', numticks=numticks))
            # axis.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: '{:.0e}'.format(x) if x in [base**i for i in range(*exponent_range, labels_skipped+1)] else ''))
            axis.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: r'${{10^{{{}}}}}$'.format(int(np.log10(x))) if x in [base**i for i in range(*exponent_range, labels_skipped+1)] else ''))


    def format_scientific_latex(self, x, precision=1, include_mantissa=True):
        exponent = int(np.floor(np.log10(x)))
        mantissa = x / 10**exponent
        if include_mantissa:
            return fr"{mantissa:.{precision}f} \times 10^{{{exponent}}}"
        else:
            return fr"10^{{{exponent}}}"


    def _subplot_label(self, axis, index, location=(0.05, 0.9), prefix="", suffix=")", 
                    fontsize=16, weight="bold", uppercase=False,
                    color="black", outline_color=None):
        """
        Add a subplot label like 'a)', optionally with color and outline for visibility.
        
        Args:
            axis: The matplotlib axis to label.
            index: The subplot index (0 = 'a', 1 = 'b', ...).
            location: (x, y) in axis coordinates.
            prefix/suffix: Optional characters like '(', ')'.
            fontsize: Font size.
            weight: Font weight (e.g., 'bold').
            uppercase: Use uppercase letters if True.
            color: Text color.
            outline: Add black/white outline for contrast if True.
        """
        # Compute label text
        letter = chr(ord("A") + index) if uppercase else chr(ord("a") + index)
        label = f"{prefix}{letter}{suffix}"
        
        # Optional outline (stroke)
        effects = [pe.withStroke(linewidth=2, foreground=outline_color)] if outline_color is not None else None

        axis.text(*location, label,
                transform=axis.transAxes,
                fontsize=fontsize,
                fontweight=weight,
                color=color,
                va="top",
                ha="left",
                path_effects=effects)



    def time_from_negative_income_to_bankruptcy(self, show_plot):
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
            bot_times, top_times = self._get_peak_pairs(i, show_plot)
            time_from_bottom_to_peak = top_times - bot_times
            
            # Store values
            time_diff_arr[i] = time_from_bottom_to_peak
            
        return np.concatenate(time_diff_arr)


    def _get_peak_pairs_single_loop(self, x, distance, width, peak_height, trough_height, peak_prominence, trough_prominence):
        """
        Loop i over peak pairs. Let n be the number of troughs between the two peaks.
        if n == 0: Add the peak with the highest x value to peak_new. If the highest peak is the left peak, only increase the right peak.
        if n == 1: Add i'th peak to peak_new and the single trough to trough_new. i += 1
        if n > 1: Add i'th peak to peak_new and the lowest trough to trough_new. i += 1
        
        Args:
            x (_type_): _description_
            distance (_type_): _description_
            width (_type_): _description_
            peak_height (_type_): _description_
            trough_height (_type_): _description_
            peak_prominence (_type_): _description_
            trough_prominence (_type_): _description_

        Returns:
            _type_: _description_
        """
        
        peak_idx, _ = find_peaks(x, distance=distance, width=width, height=peak_height, prominence=peak_prominence)
        trough_idx, _ = find_peaks(-x, distance=distance, width=width, height=trough_height, prominence=trough_prominence)
        # Check if there are any peaks, if not, return
        if np.size(peak_idx)  == 0 or np.size(trough_idx) == 0:
            print("No peaks or troughs found, returning empty arrays")
            return np.array([]), np.array([])
        
        peak_new = []
        trough_new = []
        
        i = 0        
        max_idx = len(peak_idx) - 1  # -1 such that left index is the second last element.
        while i < max_idx:
            # Get the peak pair. The right peak is the first peak chosen such that there are any troughs between the pair
            peak_left = peak_idx[i]
            n_troughs_between_pair = 0
            idx_right = i
            while n_troughs_between_pair == 0:
                idx_right += 1
                
                # Check if has reached the end, in which case return
                if idx_right == max_idx:
                    return np.array(peak_new), np.array(trough_new)
                
                peak_right = peak_idx[idx_right]
                # Count the number of troughs between the pair
                troughs_between_bool = np.logical_and(trough_idx > peak_left, trough_idx < peak_right)
                troughs_between_idx = trough_idx[troughs_between_bool]  # Transform to x-idx space
                n_troughs_between_pair = np.size(troughs_between_idx)
            # If there are more than two peaks, only keep the first
            possible_peaks = peak_idx[i:idx_right+1]  # Include the right peak thus +1
            if len(possible_peaks) > 2:            
                tallest_peak_idx = np.argmax(x[possible_peaks[:-1]])  # Do not include the rightmost peak
                peak_to_append = possible_peaks[tallest_peak_idx]
            else:
                peak_to_append = peak_left
            # Only include the lowest trough
            lowest_trough_idx = np.argmin(x[troughs_between_idx])
            lowest_trough = troughs_between_idx[lowest_trough_idx]
            # Add them to the lists
            peak_new.append(peak_to_append)
            trough_new.append(lowest_trough)
            # Update i
            i = idx_right
        
        return np.array(peak_new), np.array(trough_new)
        
        
        # i = 0        
        # max_idx = len(peak_idx) - 1
        # while i < max_idx:
        #     # Get the peak pair
        #     peak_left = peak_idx[idx_left]
        #     peak_right = peak_idx[idx_right]
        #     # Get all troughs between the two
        #     troughs_between = np.logical_and(trough_idx > peak_left, trough_idx < peak_right)  # n in the above pseudocode
        #     troughs_between_idx = trough_idx[troughs_between]
        #     n_troughs_between = np.size(troughs_between_idx)
        #     # Update based on the number of troughs between the pair
        #     if n_troughs_between == 0:
        #         if x[peak_left] > x[peak_right]:
        #             peak_idx = np.delete(peak_idx, peak_right)  # Need to remove the right peak such that we
        #             peak_new.append(peak_left)
        #             idx_right += 1
        #         else:
        #             peak_new.append(peak_right)
                    
                
        #     elif n_troughs_between == 1:
        #         peak_new.append(peak_left)
        #         trough_new.append(troughs_between_idx[0])
        #         i += 1
        #     else:  # troughs_between > 1
        #         trough_lowest = np.min(troughs_between_idx)
        #         peak_new.append(peak_left)
        #         trough_new.append(trough_lowest)
                
        #         i += 1
            # print(f"{i} / {max_i}")
        
        # return np.array(peak_new), np.array(trough_new)
                
            
    def _get_peak_pairs(self, x, distance=30, width=25, top_height=None, bot_height=None, peak_prominence=None, trough_prominence=None):
        x_min, x_max = x.min(), x.max()
        if bot_height is None: bot_height = np.abs(x_min * 0.25)
        if top_height is None: top_height = x_max * 0.75
        if peak_prominence is None: peak_prominence = 0.25 * top_height
        if trough_prominence is None: trough_prominence = 0.25 * bot_height
        top_idx, _ = find_peaks(x, height=top_height, distance=distance, prominence=peak_prominence, width=width)
        bottom_idx, _ = find_peaks(-x, height=bot_height, distance=distance, prominence=trough_prominence, width=width)  # Maybe play around with plateau size?
        
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(x, c="black")
        ax.plot(bottom_idx, x[bottom_idx], "<", c="red")
        ax.plot(top_idx, x[top_idx], "^", c="green")
        ax.grid()
        plt.show()
        
        
        # Check if there are no peaks
        if len(bottom_idx) == 0 or len(top_idx) == 0:
            print(f"{len(bottom_idx)} troughs and {len(top_idx)} peaks!")
            return np.array([]), np.array([])
        
        # Convert top indices to list to easier manipulate it        
        top_idx = top_idx.tolist()
        
        # If the first peak is a top, remove it
        if top_idx[0] < bottom_idx[0]:
            top_idx = top_idx[1:]
        
        # If the last peak is a bottom, remove it
        if bottom_idx[-1] > top_idx[-1]:
            bottom_idx = bottom_idx[:-1]
                
        # Need to check again if there are any peaks after removing end points
        if len(bottom_idx) == 0 or len(top_idx) == 0:
            print(f"{len(bottom_idx)} troughs and {len(top_idx)} peaks!")
            return np.array([]), np.array([])
        
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

            gamma1 = p * (b1**a1 / scipy.special.gamma(a1)) * x**(a1 - 1) * np.exp(-b1 * x)
            gamma2 = (1 - p) * (b2**a2 / scipy.special.gamma(a2)) * x**(a2 - 1) * np.exp(-b2 * x)
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
        
        s = self._skip_values(data)
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
    
    
    def running_KDE(self, x_data: str, bandwidth=None, eval_points=100, kernel="gaussian", data_lim=None, gname=None, time_steps_to_include=None):
        """Loop over all time steps and calculate the KDE for each time step.
        """
        # Get data
        if gname is None:
            gname = self.group_name
        self._get_data(gname)
        if x_data == "salary":
            x_data = self.s
        elif x_data == "capital":
            x_data = -self.d
        elif x_data == "workers":
            x_data = self.w
            
        data, time_steps = self._skip_values(x_data, self.time_steps)
        if time_steps_to_include is not None: time_steps = time_steps_to_include  # Option to not use the whole time series, but only time_steps_to_include
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
    
    
    # -- Power Spectral Density (PSD) analysis --
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
            freqs (np.ndarray): Array of frequency values corresponding to the PSD.
            psd (np.ndarray): Power spectral density values.
            number_of_frequencies (int): Number of dominant frequencies to return.

        Returns:
            tuple: A tuple containing two numpy arrays:
            - dominant_freqs (np.ndarray): Array of the most prominent frequency peaks.
            - dominant_powers (np.ndarray): Array of the power values of the most prominent frequency peaks.    
            - dominant_indices (np.ndarray): Array of the indices of the most prominent frequency peaks.   
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
        
        if freqs.size == 0:
            print("No data to compute PSD, returning NaN arrays")
            return np.array([np.nan]*number_of_frequencies), np.array([np.nan]*number_of_frequencies)
        
        dominant_freqs, dominant_powers = self._find_dominant_frequencies(freqs, psd, number_of_frequencies)
        
        if show_power_plot:
            plt.figure()
            plt.semilogy(freqs, psd)
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("Power Spectral Density")
            plt.title("PSD of Dataset")
            plt.show()

        return dominant_freqs, dominant_powers


    def _worker_diversity(self, gname=None):
        # Get data
        self._get_data(gname)
        # Skip values
        time, workers = self._skip_values(self.time_values, self.w)
        # Calculate the worker diversity
        diversity_arr = np.zeros(len(time))
        for t in range(len(time)):
            w_t = workers[:, t]
            diversity = np.sum(w_t) ** 2 / np.sum(w_t ** 2)
            diversity_arr[t] = diversity
        
        return time, diversity_arr
    
    
    def _return_fit(self, return_data):
        """Fits Gaussian and location transformed Student's t-distributions to the given data and returns the fitted values.
        Parameters:
            return_data (np.ndarray): The data to fit the distributions to.
        Returns:
            tuple: A tuple containing:
                - x_values (np.ndarray): The x values for plotting the fitted distributions.
                - y_gauss (np.ndarray): The y values of the fitted Gaussian distribution.
                - None: Placeholder for the Student's t-distribution (currently not used).
                - y_lst (np.ndarray): The y values of the fitted location transformed Student's t-distribution.
        """
        # Define PDFs to fit
        
        def _gaussian(x, mu, sigma):
            return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-1/2 * (x - mu)**2 / sigma**2)
        
        def _student(x, nu):
            # Coefficient in front of the PDF
            coeff = scipy.special.gamma((nu + 1.0) / 2.0) / (
                np.sqrt(nu * np.pi) * scipy.special.gamma(nu / 2.0)
            )
            # (1 + (x^2 / nu))^(-(nu+1)/2)
            inside = 1.0 + (np.power(x, 2) / nu)
            power_term = np.power(inside, -((nu + 1.0) / 2.0))
            return coeff * power_term  
        
        
        # Estimate initial values and perform minimization
        p0_list = [0.01, 0.25]  # Gaussian
        par, _ = general_functions.minimize_llh(_gaussian, return_data, p0=p0_list)

        p0_student_list = [300]  # Student t
        # par_student, _ = general_functions.minimize_llh(_student, return_data, p0=p0_student_list)
        p0_lst_list = [2.5, 0.01, 0.06]  # Location transformed student t
        par_lst, _ = general_functions.minimize_llh(self.student_t_pdf_loc_scale, return_data, p0=p0_lst_list)
        
        # Calculate x and y values for plotting
        x_values = np.linspace(np.min(return_data), np.max(return_data), 500)
        y_gauss = _gaussian(x_values, *par)
        # y_student = _student(x_values, *par_student)
        y_lst = self.student_t_pdf_loc_scale(x_values, *par_lst)
        
        # Print the fitted values for Gaussian
        mu1, sigma1 = par
        print(f"Gaussian: mu = {mu1:.2f}, sigma = {sigma1:.2f}, ")
        
        # Print the fitted values for student t 
        # nu_student = par_student[0]
        # print(f"Student t: nu = {nu_student:.2f}")
        
        # Print the fitted values for location transformed student t
        nu_lst, mu_lst, sigma_lst = par_lst
        print(f"Location transformed Student t: nu = {nu_lst:.2f}, mu = {mu_lst:.2f}, sigma = {sigma_lst:.2f}")
        
        return x_values, y_gauss, None, y_lst


    def student_t_pdf_loc_scale(self, x, nu, mu, sigma):
        """
        Computes the PDF of the location-scale Student's t-distribution at value x,
        with degrees of freedom nu, location mu, and scale sigma.

        Returns
        -------
        pdf : float or numpy.ndarray
            The value(s) of the Student's t-distribution PDF at x with the
            specified location and scale. The shape matches the shape of x
            if x is an array.
        """
        # Ensure sigma is positive
        if sigma <= 0:
            raise ValueError("Scale parameter sigma must be positive.")

        # Coefficient in front of the PDF
        coeff = scipy.special.gamma((nu + 1.) / 2.) / (
            np.sqrt(nu * np.pi) * scipy.special.gamma(nu / 2.) * sigma
        )

        # Shift and scale transformation: (x - mu) / sigma
        z = (x - mu) / sigma
        
        # Core term: (1 + (z^2 / nu))^( - (nu + 1) / 2 )
        inside = 1.0 + (z**2) / nu
        power_term = inside ** ( - (nu + 1.) / 2. )
        
        return coeff * power_term
    
    
    def _asset_return(self, data_name:str, time_period:int):
        """_summary_

        Args:
            data_name (str): What variable to use for the return calculation. Must be "mu", "capital_sum" or "capital_individual_mean", capital_individual_all.
            time_period (int): Shift in time for the return calculation, equal to tau in: log(p(t + tau)) - log(p(t)).

        Returns:
            _type_: _description_
        """
        # Get data
        if data_name == "mu":
            data = self.mu
        elif data_name == "capital_sum":
            capital = -self.d
            data = np.sum(capital, axis=0)
        elif data_name == "capital_individual_mean":
            # Individual capital has an extra dimension and the calculation must be done seperately
            data = -self.d
            data = self._skip_values(data)
            data[data < 1e-10] = 1e-10
            r_individual = np.log(data[:, time_period:]) - np.log(data[:, :-time_period])
            r = np.mean(r_individual, axis=0)
            return r
        elif data_name == "capital_individual_all":
            # Individual capital has an extra dimension and the calculation must be done seperately
            data = -self.d
            data = self._skip_values(data)
            # Do not calculate the return of values below m
            valid_mask = (data[:, :-1] >= self.m) & (data[:, 1:] >= self.m)  # Neighbouring columns must both be above m to calculate the difference
            relative_diff = np.full((data.shape[0], data.shape[1]-1), np.nan)  # Create an empty array to be filled only the valid places
            relative_diff[valid_mask] = np.log(data[:, time_period:][valid_mask]) - np.log(data[:, :-time_period][valid_mask])
                        
            r = np.ravel(relative_diff)
            return r
        else:
            print(f"{data_name} is an invalid data_name, must be 'mu', 'capital_sum', 'capital_individual_all' or 'capital_individual_mean'")
            return np.array([])

        # Skip values
        data = self._skip_values(data)
        # Replace 0 values with a very small number to avoid division by 0
        data[data < 1e-10] = 1e-10
        # Calculate the return
        r = np.log(data[time_period:]) - np.log(data[:-time_period])
        return r


    def _load_multiple_return_data(self, group_name_list):
        """Given a set of parameters, load the corresponding data and return it.
        """
        r_arr = np.zeros((len(group_name_list), self.time_steps - self.skip_values))
        
        for i, gname in enumerate(group_name_list):
            # Load data and store it in arrays
            self._get_data(gname)
            r = self._asset_return(data_name="capital_indiviual_mean", time_period=1)            
            r_arr[i, : ] = r
        
        return r_arr
    
    
    def onedim_KDE(self, x, bandwidth, x_eval=None, N_eval_points=100, kernel="gaussian", log_scale=False):
        # If not given specific evaluation points, evaluate the whole space
        if x_eval is None:
            x_min, x_max = x.min(), x.max()
            if log_scale:
                if x_min <= 0: raise ValueError(f"Data must be positive for log scale, but xmin is {x_min}")
                x_eval = np.logspace(np.log10(x_min), np.log10(x_max), N_eval_points)
            else:
                x_eval = np.linspace(x_min, x_max, N_eval_points)
        # KDE
        kde = KernelDensity(bandwidth=bandwidth, kernel=kernel)
        kde.fit(x[:, None])
        P = np.exp(kde.score_samples(x_eval[:, None]))
        return x_eval, P
    
    
    def _detect_recession_simple(self, window_size, peak_distance=15, peak_width=5, peak_height=None, trough_height=None, plot=False):
        """Using find_peaks, finds the time of the recessions and their durations.
        """
        # Get data
        self._get_data(self.group_name)
        mu = self._skip_values(self.mu)

        # Perform rolling average and divide by W
        mu_rolling_averaged = uniform_filter1d(mu, size=window_size) / self.W
                
        # Find peaks
        mu_min = mu_rolling_averaged.min()
        mu_max = mu_rolling_averaged.max()

        if peak_height is None:
            peak_height = mu_min + (mu_max - mu_min) / 4
        peaks, peak_info = find_peaks(mu_rolling_averaged, height=peak_height, distance=peak_distance, width=peak_width)
        
        # Find troughs
        if trough_height is None:
            trough_height = mu_max - (mu_max - mu_min) / 4
        
        troughs, trough_info = find_peaks(-mu_rolling_averaged, height=-peak_height, width=peak_width)
        
        if plot:
            fig, ax = plt.subplots(figsize=(10, 7.5))
            ax.plot(mu_rolling_averaged)
            ax.plot(peaks, mu_rolling_averaged[peaks], "^")
            ax.plot(troughs, mu_rolling_averaged[troughs], ">")
            ax.set(xlabel="Time", ylabel="Price", title=r"Peaks in $\mu/W$")
            plt.show()
        
        return peaks, troughs
    
    
    def _detect_recession_single_loop(self, window_size, peak_distance, peak_width, peak_height, trough_height, peak_prominence, trough_prominence, plot=False):
        """Smooth (rolling mean) the mu data and find peak pairs in it.
        """
        # Load data
        self._get_data(self.group_name)
        mu, time = self._skip_values(self.mu, self.time_values)
        
        # Smooth mu and divide by W
        mu_smooth = uniform_filter1d(mu, size=window_size) / self.W
        
        # Find the bottom of the smoothed mu curve using find_peaks. The distance between the peak and the trough is the recession duration
        peak_times, trough_times = self._get_peak_pairs_single_loop(mu_smooth, peak_distance, peak_width, 
                                                    peak_height=peak_height, trough_height=trough_height, peak_prominence=peak_prominence, 
                                                    trough_prominence=trough_prominence)
        if plot:
            fig, ax = plt.subplots(figsize=(16, 8), dpi=300)
            ax.plot(time, mu_smooth, c="black")
            ax.plot(trough_times+self.skip_values, mu_smooth[trough_times], "<", c="red")
            ax.plot(peak_times+self.skip_values, mu_smooth[peak_times], "^", c="green")
            ax.set(xlabel="Time", ylabel=r"$\mu$", title=r"Detecting peaks in $\mu$")
            ax.grid()
            plt.show()
        
        return trough_times, peak_times
    
    
    def _detect_recession(self, window_size, peak_distance, peak_width, peak_height, trough_height, peak_prominence, trough_prominence, plot=False):
        """Smooth (rolling mean) the mu data and find peak pairs in it.
        """
        # Load data
        self._get_data(self.group_name)
        mu, time = self._skip_values(self.mu, self.time_values)
        
        # Smooth mu and divide by W
        mu_smooth = uniform_filter1d(mu, size=window_size) / self.W
        
        # Find the bottom of the smoothed mu curve using find_peaks. The distance between the peak and the trough is the recession duration
        trough_times, peak_times = self._get_peak_pairs(mu_smooth, peak_distance, peak_width, 
                                                    top_height=peak_height, bot_height=trough_height, peak_prominence=peak_prominence, 
                                                    trough_prominence=trough_prominence)
        trough_times = trough_times[1:]
        peak_times = peak_times[:-1]
        
        if plot:
            fig, ax = plt.subplots(figsize=(16, 8), dpi=300)
            ax.plot(time, mu_smooth, c="black")
            ax.plot(trough_times+self.skip_values, mu_smooth[trough_times], "<", c="red")
            ax.plot(peak_times+self.skip_values, mu_smooth[peak_times], "^", c="green")
            ax.set(xlabel="Time", ylabel=r"$\mu$", title=r"Detecting peaks in $\mu$")
            ax.grid()
            plt.show()
        
        return trough_times, peak_times
    
    
    def _recession_time_between_and_duration(self, window_size, peak_distance, peak_width, peak_height, trough_height, peak_prominence, trough_prominence, plot=False, return_peaks=False):
        """
        Get the times of the troughs and the peaks
        Calculate the duraction of the recessions and the time between the recessions
        Duration is time from peak to trough and period is the time between peaks
        Args:
            window_size (int, optional): _description_. Defaults to 10.
            peak_distance (int, optional): _description_. Defaults to 15.
            peak_width (int, optional): _description_. Defaults to 5.
            peak_height (_type_, optional): _description_. Defaults to None.
            trough_height (_type_, optional): _description_. Defaults to None.
            peak_prominence (_type_, optional): _description_. Defaults to None.
            trough_prominence (_type_, optional): _description_. Defaults to None.
            plot (bool, optional): _description_. Defaults to False.

        Returns:
            tupple (array, array): period, duration
        """
        # Get the times of the troughs and the peaks
        troughs, peaks = self._detect_recession_single_loop(window_size, peak_distance, peak_width, peak_height, trough_height, peak_prominence, trough_prominence, plot)
        
        # Calculate the duraction of the recessions and their frequency.
        # Duration is time from peak to trough and period is time between peaks
        time_between = np.diff(peaks)
        duration = troughs - peaks

        print("Total number of peak pairs: ", np.size(peaks))        
        negative_durations_positions = np.where(duration < 0)[0]
        number_of_negative_durations = np.size(negative_durations_positions)
        if number_of_negative_durations > 0:
            print(f"Warning: Detected {number_of_negative_durations} negative durations at indices:")
            print(negative_durations_positions)
        
        if return_peaks:
            return time_between, duration, troughs, peaks
        
        return time_between, duration
    
    
    def _save_recession_results(self, peak_kwargs):
        time_between, duration, troughs, peaks = self._recession_time_between_and_duration(**peak_kwargs, return_peaks=True)
        filename = dir_path_output / f"recession_results_{self.group_name}.npz"
        np.savez(file=filename, time_between=time_between, duration=duration, troughs=troughs, peaks=peaks)
    
    
    def _load_recession_results(self):
        filename = dir_path_output / f"recession_results_{self.group_name}.npz"
        try:
            with np.load(filename) as data:
                time_between = data["time_between"]
                duration = data["duration"]
                troughs = data["troughs"]
                peaks = data["peaks"]
                return time_between, duration, troughs, peaks
        except FileNotFoundError:
            print(f"Error: No data found for {self.group_name}. Returning empty arrays")
            print("Did you forget to run _save_recession_results?")
            return np.array([]), np.array([]), np.array([]), np.array([])
    
    
    def _load_lifespan_data(self):
        """Load the company mortality data read from https://royalsocietypublishing.org/cms/asset/fabe4e8a-b0a6-47ae-a333-6f4c3753dc51/rsif20150120f02.jpg

        Returns:
            (lifespan, counts): How long companies live and the number of companies at that lifetime
        """
        name = "Final_Cleaned_and_Calibrated_Data.csv"
        file_path = dir_path_data / name  
        data = np.genfromtxt(file_path, delimiter=",")
        lifespan = data[:, 0]
        counts = data[:, 1]
        return lifespan, counts


    def _load_peak_trough_data(self):
        file_name = "20210719_cycle_dates_pasted.csv"
        file_path = dir_path_data / file_name
        df = pd.read_csv(file_path, delimiter=",")

        # Remove rows with missing data
        df.dropna(axis=0, inplace=True)

        # Convert the 'peak' and 'trough' columns to datetime
        df["peak"] = pd.to_datetime(df["peak"])
        df["trough"] = pd.to_datetime(df["trough"])
        
        return df        

        
    def _load_recession_data(self, separate_post_war=False):
        """Load and preprocess NBER recession data

        Args:
            separate_post_war (bool): If True, return both all data and post-WWII data separately.

        Returns:
            If separate_post_war is False:
                (time_between, duration): All recession data.
            If separate_post_war is True:
                ((time_between_all, duration_all), (time_between_postwar, duration_postwar))
        """
        df = self._load_peak_trough_data()

        # Calculate duration of each recession
        df["duration"] = (df["trough"] - df["peak"]).dt.days

        # Time between consecutive recessions (peaks)
        df["time_between"] = df["peak"].diff().dt.days

        if not separate_post_war:
            return df["time_between"].dropna(), df["duration"]
        else:
            postwar_start = pd.Timestamp("1946-01-01")
            df_post = df[df["peak"] >= postwar_start].copy()

            return (
                (df["time_between"].dropna(), df["duration"]),
                (df_post["time_between"].dropna(), df_post["duration"]),
            )
    

    def _load_company_size_data(self, interval=True):
        """Load and preprocess the CENSUR company size data.

        Returns:
            (labels, counts): Company size intervals and the number of companies in the intervals
        """
        file_name = "us_state_naics_detailedsizes_2021.csv"
        file_path = dir_path_data / file_name
        df = pd.read_csv(file_path, delimiter=",")

        # Filter to 'Total' sector
        df = df[
            (df["NAICSDSCR"] == "Total") &
            (df["STATEDSCR"] == "United States") &
            (df["NAICS"] == "--") &
            (df["STATE"] == 0)
        ]
        # Remove the rows with: "<20", "<500", "Total"
        labels_not_allowed = ["06: <20", "19: <500", "01: Total"]
        labels = []
        counts = []
        for _, row in df.iterrows():
            label = row["ENTRSIZEDSCR"]
            count = row["FIRM"]
            
            if label in labels_not_allowed:
                continue
            
            # Structure is e.g. 08: 25-29, 02: <5
            # For all: remove the first 3 entries, strip whitespace
            # Then take care of the special case of <5
            label = label[3:].strip()
            if not interval:
                label = label.split("-")[0] 
            label = label.replace(",", "")  # Large numbers are written as e.g. 5,000

            labels.append(label)
            counts.append(count)
        
        return labels, counts


    def _prepare_firm_size_pmf(self, upper_limit=10_000, normalize=True):
        """
        Maps irregular firm size bins to power-of-3 width bins, computes
        adjusted frequencies (count / bin width), and returns geometric bin centers.
        
        Parameters:
            labels (list of str): firm size bin labels (e.g., "5-9", "<5", "5,000+")
            counts (list of int): firm counts in each bin
        
        Returns:
            centers (np.ndarray): geometric mean of bin edges
            pmf (np.ndarray): adjusted frequency (counts / width)
        """
        # Get data
        labels, counts = self._load_company_size_data(interval=True)
        
        # Define power-of-3 bin edges: [1, 4, 10, 28, 82, 244, 730, ...]
        bin_edges = [1]
        while bin_edges[-1] < upper_limit:
            bin_edges.append(bin_edges[-1] * 3)

        # Initialize count accumulator per bin
        binned_counts = np.zeros(len(bin_edges) - 1)

        # Helper: convert label to numeric range
        def parse_label(label):
            label = label.replace(",", "")
            if "-" in label:
                low, high = map(int, label.split("-"))
            elif label.startswith("<"):
                low = 0
                high = int(label[1:])
            elif label.endswith("+"):
                low = int(label[:-1])
                high = upper_limit  # assume an arbitrary upper bound
            else:
                return None
            return low, high

        # Bin the original counts
        for label, count in zip(labels, counts):
            bounds = parse_label(label)
            if not bounds:
                continue
            low, high = bounds

            # Distribute count across all power-of-3 bins that intersect the label
            for i in range(len(bin_edges) - 1):
                b_low = bin_edges[i]
                b_high = bin_edges[i + 1]
                overlap_low = max(low, b_low)
                overlap_high = min(high, b_high)
                overlap = max(0, overlap_high - overlap_low)
                width = high - low
                if overlap > 0 and width > 0:
                    # Proportionally assign part of the count to this bin
                    binned_counts[i] += count * (overlap / width)

        # Compute adjusted frequency and geometric bin center
        centers = np.sqrt(np.array(bin_edges[:-1]) * np.array(bin_edges[1:]))
        widths = np.diff(bin_edges)
        pmf = binned_counts / widths
        
        if normalize:
            area = np.sum(pmf * widths)
            pmf /= area  # normalize so area = 1

        return centers, pmf

    
    def _get_lifespan(self, gname=None):
        """For each company, find the number of time steps between two C=0 events i.e. birth to death
        """
        # Get data
        self._get_data(gname)
        C = self._skip_values(-self.d)
        
        lifespan_all = []
        
        # Loop over companies        
        for i in range(self.N):
            C_comp = C[i, :]
            C_is_0 = np.where(C_comp == 0)
            life_spans = np.diff(C_is_0)
            life_spans = life_spans[life_spans>1]  # If the company did not change its capital while at 0
            lifespan_all.append(life_spans)
                
        return np.concatenate(lifespan_all)
    
    
    def _get_sp500_constituents(self):
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        df = tables[0]  # First table is the one you want
        return df["Symbol"].tolist()


    def _save_sp500_asset_return(self, change_days=10, batch_size=10):

        tickers = self._get_sp500_constituents()
        all_prices = []

        print(f"Downloading {len(tickers)} tickers in batches of {batch_size}...")

        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            try:
                prices = yf.download(batch, start="2000-01-01", end="2025-04-01")["Close"]
                all_prices.append(prices)
            except Exception as e:
                print(f"Batch {i//batch_size} failed: {e}")
            time.sleep(10)  # delay to avoid being rate-limited

        # Combine and clean
        all_prices_df = pd.concat(all_prices, axis=1)
        valid_prices = all_prices_df.dropna(axis=1, thresh=0.9 * len(all_prices_df))

        # Compute returns
        log_returns = np.log(valid_prices / valid_prices.shift(change_days)).dropna()

        # Save
        filename = dir_path_data / f"asset_return_sp500_changedays{change_days}.csv"
        log_returns.to_csv(filename)
        print(f"Saved {log_returns.shape[1]} tickers' returns.")


    def _load_sp500_asset_return(self, change_days=10):
        """Load individual companies' asset return

        Args:
            change_days (int, optional): _description_. Defaults to 10.

        Returns:
            np.ndarray: Log returns over change_days time
        """
        filename = dir_path_data / f"asset_return_sp500_changedays{change_days}.csv"
        try:
            log_returns = pd.read_csv(filename, index_col=0, parse_dates=True)
        except FileNotFoundError:
            print("No Yfinance data for S&P 500 found. Saving data now")
            self._save_sp500_asset_return(change_days=change_days)
            log_returns = pd.read_csv(filename, index_col=0, parse_dates=True)
        return np.ravel(log_returns)
    
    
    def duration_spread_over_mean(self, peak_kwargs):
        """Compare the spread/mean for the recession duration of our model and the data
        """
        # Get model and data
        time, duration, _, _ = self._load_recession_results() # _recession_time_between_and_duration(plot=False, **peak_kwargs)
        time_NBER, duration_NBER = self._load_recession_data()
        
        # Calculate spread / mean
        spread_over_mean = np.std(duration) / np.mean(duration)
        spread_over_mean_NBER = np.std(duration_NBER) / np.mean(duration_NBER)
        
        som_time = np.std(time) / np.mean(time)
        som_time_NBER = np.std(time_NBER) / np.mean(time_NBER)
        
        return spread_over_mean, som_time, spread_over_mean_NBER, som_time_NBER
    

    def bootstrap_median_ci(self, data, n_bootstrap=1000, ci=0.68):
        """Bootstrap estimate of the median and its error."""
        medians = np.median(np.random.choice(data, size=(n_bootstrap, len(data)), replace=True), axis=1)
        lower = np.percentile(medians, (1 - ci) / 2 * 100)
        upper = np.percentile(medians, (1 + ci) / 2 * 100)
        err = (upper - lower) / 2
        return np.median(data), err


    def time_scale_of_the_system(self):
        """Compare timescales in model and data (recession time, duration, company lifespan)."""
        # Load data
        tb_model, dur_model, _, _ = self._load_recession_results()
        tb_data, dur_data = self._load_recession_data()
        life_model = self._get_lifespan()
        life_data_x, life_data_logy = self._load_lifespan_data()
        # Skip first point in life_data, as it is an outlier
        life_data_x = life_data_x[1:]
        life_data_logy = life_data_logy[1:]
        # Convert NBEr to years
        tb_data /= 365.25
        dur_data /= 365.25
        
        results = []

        # --- Compare medians of tb and dur ---
        for label, model, data in [("Time between recessions", tb_model, tb_data),
                                ("Recession duration", dur_model, dur_data)]:
            median_model, err_model = self.bootstrap_median_ci(model)
            median_data, err_data = self.bootstrap_median_ci(data)

            diff = median_model - median_data
            diff_err = np.sqrt(err_model**2 + err_data**2)

            ratio = median_model / median_data
            ratio_err = ratio * np.sqrt((err_model/median_model)**2 + (err_data/median_data)**2)

            results.append({
                "Metric": label,
                "Model": f"${median_model:.2f} \\pm {err_model:.2f}$",
                "Data": f"${median_data:.2f} \\pm {err_data:.2f}$",
                "Ratio": f"${ratio:.2f} \\pm {ratio_err:.2f}$"
            })

        # --- Lifespan half-time from CDF = 0.5 (assume exponential tail) ---
        half_life_model = np.median(life_model)
        tau_model = half_life_model / np.log(2)

        counts = 10 ** life_data_logy
        pmf = counts / np.sum(counts)      # Normalize to get PMF
        cdf = np.cumsum(pmf)               # Compute CDF
        # Find the first index where CDF >= 0.5
        half_idx = np.argmax(cdf >= 0.5)
        half_life_data = life_data_x[half_idx]
        tau_data = half_life_data / np.log(2)

        tau_ratio = tau_model / tau_data

        results.append({
            "Metric": "Lifespan half-time",
            "Model": f"${half_life_model:.2f}$",
            "Data": f"${half_life_data:.2f}$",
            "Ratio": f"${tau_ratio:.2f}$"
        })

        df = pd.DataFrame(results)
        return df

    
    def _save_inflation_data(self, source="CPI", start="1959-01-01", end="2025-04-01"):
        tickers = {
            "CPI": "CPIAUCSL",     # Consumer Price Index
            "PCE": "PCEPI"         # Personal Consumption Expenditures Price Index
        }

        if source.upper() not in tickers:
            raise ValueError("Source must be either 'CPI' or 'PCE'.")
        # Download data
        series = web.DataReader(tickers[source.upper()], "fred", start, end)
        name = source + f"_inflation_data.pkl"
        filename = dir_path_output / name
        series.to_pickle(filename)
        return series
    

    def _load_inflation_data(self, source="CPI", start="1959-01-01", end="2025-03-01", freq="ME", change_type="log", annualized=False):
        """
        Download CPI or PCE data from FRED via yfinance and calculate inflation.

        Args:
            source (str): "CPI" or "PCE".
            start (str): Start date in "YYYY-MM-DD" format.
            end (str): End date in "YYYY-MM-DD" format.
            freq (str): Frequency for resampling: "M" (monthly), "Q" (quarterly), etc.
            change_type (str): "log" for log returns, "percent" for percentage change.
            annualized (bool): Whether to annualize the inflation rate.

        Returns:
            pd.DataFrame: DataFrame with the price index and computed inflation.
        """
        tickers = {
            "CPI": "CPIAUCSL",     # Consumer Price Index
            "PCE": "PCEPI"         # Personal Consumption Expenditures Price Index
        }

        if source.upper() not in tickers:
            raise ValueError("Source must be either 'CPI' or 'PCE'.")
        if change_type.lower() not in {"log", "percent"}:
            raise ValueError("change_type must be 'log' or 'percent'.")

        # Load data
        name = source + f"_inflation_data.pkl"
        filename = dir_path_output / name
        try:
            series = pd.read_pickle(filename)
        except FileNotFoundError:
            print(f"File {name} not found. Downloading data now")
            series = self._save_inflation_data(source, start, end)

        series.name = source.upper()
        series = series.resample(freq).mean()  # Take NaNs into account

        # Determine periods per year
        freq_letter = freq[0].upper()
        periods_per_year = {"A": 1, "Y": 1, "Q": 4, "M": 12}.get(freq_letter, 12)

        # Compute inflation
        if change_type.lower() == "log":
            raw = np.log(series / series.shift(1))
            if annualized:
                inflation = raw * periods_per_year
                inflation.name = "annual_log_inflation"
            else:
                inflation = raw
                inflation.name = "log_inflation"
        else:
            raw = (series / series.shift(1) - 1) * 100
            if annualized:
                inflation = ((series / series.shift(1)) ** periods_per_year - 1) * 100
                inflation.name = "annual_percent_inflation"
            else:
                inflation = raw
                inflation.name = "percent_inflation"

        return series, inflation.dropna()

    
    def _get_inflation(self, change_type="log", window_size=1):
        """Find the price change (log or procent determined by change_type) of either mu or s depending on price_change_variable.

        Args:
            change_type (str, optional): _description_. Defaults to "log".
            window_size (int, optional): _description_. Defaults to 1.
            price_change_variable (str, optional): _description_. Defaults to "mu".

        Returns:
            _type_: _description_
        """
        # Load data
        self._get_data(self.group_name)
        mu, w_paid = self._skip_values(self.mu, self.w_paid)
        average_wage_paid = mu / w_paid
        variable = uniform_filter1d(average_wage_paid, size=window_size)
            
        # Compute inflation
        if change_type.lower() == "log":
            inflation = np.log10(variable[1:] / variable[:-1])
        elif change_type.lower() == "percent":
            inflation = (variable[1:] / variable[:-1] - 1) * 100    
        else:
            print(f"{change_type} is not a valid change_type")
        
        return variable, inflation
    
    
    def _save_N_W_results(self, gname_arr):
        """Calculate the following, based on input:
            1. Spread / Mean each time step, then mean that. 
            2. Median lifespan
        """
        dimensions = gname_arr.shape
        diversity_arr = np.empty_like(gname_arr, dtype=np.float32)
        median_lifespan_arr = np.empty_like(diversity_arr)
        N_arr = np.empty(dimensions[0], dtype=np.int32) 
        ratio_arr = np.empty(dimensions[1], dtype=np.int32)
        
        for i in range(dimensions[0]):
            for j in range(dimensions[1]):
                lifespan = self._get_lifespan(gname_arr[i, j])
                median_lifespan = np.median(lifespan)
                _, D = self._worker_diversity(gname_arr[i, j])
                D_mean = np.mean(D) / self.N
                # Store values
                diversity_arr[i, j] = D_mean
                median_lifespan_arr[i, j] = median_lifespan
                ratio_arr[j] = int(self.W / self.N)  # Is the same for all i, so will overwrite.
            N_arr[i] = self.N * 1

        filename = dir_path_output / f"N_W_results_{self.group_name}.npz"
        np.savez(filename, N=N_arr, ratio=ratio_arr, D=diversity_arr, lifespan=median_lifespan_arr)
        return N_arr, ratio_arr, diversity_arr, median_lifespan_arr


    def _load_N_W_results(self, gname_arr):
        filename = dir_path_output / f"N_W_results_{self.group_name}.npz"
        try:
            with np.load(filename) as data:
                N = data["N"]
                ratio = data["ratio"]
                D = data["D"]
                lifespan = data["lifespan"]
                return N, ratio, D, lifespan
        except FileNotFoundError:
            print(f"No N_W data found for {self.group_name}. Creating data now.")
            N_arr, ratio_arr, diversity_arr, median_lifespan_arr = self._save_N_W_results(gname_arr)
            return N_arr, ratio_arr, diversity_arr, median_lifespan_arr
        
    
    def _save_KDE(self, gname, KDE_par: dict):
        # KDE
        s_eval, KDE_prob = self.running_KDE("salary", **KDE_par, gname=gname)  # KDE probabilities
        filename = dir_path_output / f"KDE_{gname}.npz"
        np.savez(filename, s_eval=s_eval, KDE_prob=KDE_prob)
        return s_eval, KDE_prob


    def _load_KDE(self, gname, KDE_par: dict):
        filename = dir_path_output / f"KDE_{gname}.npz"
        try:
            with np.load(filename) as data:
                s_eval = data["s_eval"]
                KDE_prob = data["KDE_prob"]
        except FileNotFoundError:
            print(f"No KDE data found for {gname}. Creating data now.")
            s_eval, KDE_prob = self._save_KDE(gname, KDE_par)
        return s_eval, KDE_prob
        

    def generate_timescale_latex_tables_columnwise(self, result_list, param_list, metric_names=["Time between recessions", "Recession duration", "Lifespan half-time"]):
        """
        Generate two LaTeX tables: one for absolute values and one for ratios.
        Rows: metrics, Columns: parameters (including 'Data').
        """
        formatted_param = [f"$\\alpha={a}$" for a in param_list]

        # Prepare data storage
        model_dict = {metric: [] for metric in metric_names}
        ratio_dict = {metric: [] for metric in metric_names}

        # Fill model values and ratios
        for df in result_list:
            for metric in metric_names:
                row = df[df["Metric"] == metric].iloc[0]
                model_dict[metric].append(row["Model"])
                ratio_dict[metric].append(row["Ratio"])

        # Extract one copy of data values
        data_df = result_list[0]
        data_col = [data_df[data_df["Metric"] == m].iloc[0]["Data"] for m in metric_names]

        # Add 'Data' as first column
        model_table = pd.DataFrame(model_dict, index=formatted_param).T
        model_table.insert(0, "Data", data_col)

        ratio_table = pd.DataFrame(ratio_dict, index=metric_names, columns=formatted_param)

        model_latex = model_table.to_latex(escape=False, column_format="l" + "c" * model_table.shape[1], caption="Model and data timescales for each metric.", label="tab:timescales_columnwise")
        ratio_latex = ratio_table.to_latex(escape=False, column_format="l" + "c" * ratio_table.shape[1], caption="Ratio of model to data timescales for each metric.", label="tab:ratios_columnwise")

        return model_latex, ratio_latex