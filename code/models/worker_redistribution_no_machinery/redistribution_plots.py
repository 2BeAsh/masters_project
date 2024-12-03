import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, LogNorm
from pathlib import Path
import h5py
import functools
import matplotlib.animation as animation

from scipy.signal import find_peaks

# My files
import general_functions
from redistribution_no_m_master import dir_path_output, dir_path_image, group_name, ds_space, rf_space


class BankVisualization(general_functions.PlotMethods):
    def __init__(self, group_name, show_plots, add_parameter_text_to_plot):
        super().__init__(group_name)
        self.group_name = group_name
        self.T_func_name = self.group_name.split("_")[-1].replace("_", "")
        self.show_plots = show_plots
        self.add_parameter_text_to_plot = add_parameter_text_to_plot
        
        # Check if the path to the image folder exists, otherwise create it
        dir_path_image.mkdir(parents=True, exist_ok=True)
        self.dir_path_image = dir_path_image
        self.data_path = dir_path_output / "redistribution_no_m_simulation_data.h5"
        
        # Load data        
        with h5py.File(self.data_path, "r") as file:
            data_group = file[self.group_name]
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
            
            # Peaks
            try:
                self.peak_idx = data_group["peak_idx"][:]
                self.peak_vals = data_group["peak_vals"][:]
            except KeyError:
                print("No peak data found in main file")
                self.peak_idx = None
                self.peak_vals = None
        
            # Attributes
            self.ds = data_group.attrs["salary_increase"]
            self.rf = data_group.attrs["interest_rate_free"]
            self.W = data_group.attrs["W"]
            
        self.N = self.w.shape[0]
        self.time_steps = self.w.shape[1]
        self.time_values = np.arange(self.time_steps)

    
    def _get_group_name(self) -> str:
        return f"Steps{self.time_steps}_N{self.N}_W{self.W}_ds{self.ds:.3f}_rf{self.rf:.3f}_" + self.T_func_name

    
    def plot_companies(self, N_plot):
        w_plot = self.w[: N_plot, :].T
        d_plot = self.d[: N_plot, :].T
        
        fig, (ax, ax1) = plt.subplots(nrows=2)

        ax.plot(self.time_values, w_plot)
        ax.set(ylabel="Log Number of workers", title="Workforce", xlim=(0, self.time_steps), yscale="log")
        ax.grid()

        ax1.plot(self.time_values, d_plot)
        ax1.set(ylabel="Log Price", title="Debt", xlabel="Time", xlim=(0, self.time_steps), yscale="symlog")
        ax1.grid()
        
        # Display parameter values
        if self.add_parameter_text_to_plot: self._add_parameters_text(ax)
        
        # Save figure
        self._save_fig(fig, "singlecompanies")
        if self.show_plots: plt.show()
        
        
    def plot_means(self):
        w_mean = np.mean(self.w, axis=0)
        d_mean = np.mean(self.d, axis=0)
        
        # Mask for debt 
        pos_mask = d_mean >= 0
        neg_mask = d_mean < 0
        
        # Create figure
        fig, ax = plt.subplots()
        
        # Debt
        ax.plot(self.time_values[pos_mask], d_mean[pos_mask], ".", label=r"$d>0$", c="green")
        ax.plot(self.time_values[neg_mask], np.abs(d_mean[neg_mask]), ".", label=r"$\|d<0\|$", c="red")
        
        # Production
        ax.plot(self.time_values, w_mean, label=r"$w$")

        # Setup
        ax.set(xlabel="Time", ylabel="Log Price", title="Mean values", yscale="log", xlim=(0, self.time_steps))
        ax.grid()
        self._add_legend(ax, ncols=4, y=0.9)
        
        # Display parameters
        if self.add_parameter_text_to_plot: self._add_parameters_text(ax)
        # Save and show
        self._save_fig(fig, "means")
        if self.show_plots: plt.show()
                
        
    def plot_interest_rates(self):
        """Plot the interest rate and the fraction of companies gone bankrupt over time.
        """
        fig, (ax, ax1) = plt.subplots(nrows=2)

        # ax Interest rate and free interest rate
        ax.axhline(y=0.05, ls="--", label="Interest rate free")
        ax.plot(self.time_values, self.interest_rate, label="Interest rate", c="firebrick")
        ax.set(ylabel="Interest rate", xlim=(0, self.time_steps))
        ax.grid()
        ax.legend(bbox_to_anchor=(0.5, 1), loc="lower center", fontsize=10, ncols=3)
        
        # ax1 bankruptcies
        ax1.plot(self.time_values, self.went_bankrupt / self.N, label="Bankruptcies", c="darkorange")
        ax1.set(ylabel="Bankrupt fraction", title="Bankruptcies", xlim=(0, self.time_steps))
        ax1.grid()
        
        # Add the peaks to the bankruptcies
        if self.peak_idx is not None:
            ax1.plot(self.peak_idx, self.peak_vals, "x", label="Peaks", c="black")
        
        # Add legend
        self._add_legend(ax1, x=0.1, y=0.8, ncols=2, fontsize=8)
        # Add parameters text
        if self.add_parameter_text_to_plot: self._add_parameters_text(ax)
        # Save and show
        self._save_fig(fig, "interest_rates")
        if self.show_plots: plt.show()
        
        
    def size_distribution(self):
        # Log histogram of p values, and a histogram of d values where positive and abs(negative) are plotted seperately
        # Final time value of production and debt
        production_final = self.w[:, -1]  # Production minimum value is 1
        debt_final = self.d[:, -1] + 1e-6  # Prevent zero values
        debt_positive = debt_final[debt_final > 0]
        debt_negative_abs = np.abs(debt_final[debt_final < 0])

        # Binning
        Nbins = int(np.sqrt(self.time_steps)) 
        bins_p = 10 ** np.linspace(np.log10(1e0), np.log10(production_final.max() * 10), Nbins)  # Log x cuts off large values if max range value is not increased
        bins_d = 10 ** np.linspace(np.log10(1e-6), np.log10(np.abs(debt_final).max() * 10), Nbins)
        
        # Create figure
        fig, (ax, ax1) = plt.subplots(ncols=2)
        
        # ax: Production
        counts_p, _, _ = ax.hist(production_final, bins=bins_p)
        
        # ax 1: Debt
        # Plot positive and negative values separately
        counts_d_pos, _, _ = ax1.hist(debt_positive, bins=bins_d, label="Positive debt", color="green", alpha=0.7)
        counts_d_neg, _, _ = ax1.hist(debt_negative_abs, bins=bins_d, label="abs negative debt", color="red", alpha=0.7)

        # Setup
        ylim = (0, np.max((counts_p, counts_d_neg, counts_d_pos+1)))
        ax.set(xlabel="Production", ylabel="Counts", title="Production values at final time", xscale="log", ylim=ylim)
        ax1.set(xlabel="Debt", title="Debt values at final time", xscale="log", ylim=ylim)
        self._add_legend(ax1, ncols=2, y=0.9, fontsize=8)
        ax.grid()
        ax1.grid()
        
        # Parameters text
        if self.add_parameter_text_to_plot:  self._add_parameters_text(ax)
        
        # Save and show figure
        self._save_fig(fig, "hist")        
        if self.show_plots: plt.show()
                
        
    def plot_production_capacity(self):
        w_final = self.w[:, -1]
        max_val = np.max(w_final)
        bins = np.arange(0, max_val + 1 , 1)
        
        fig, ax  = plt.subplots()
        counts, edges, _ = ax.hist(w_final, bins=bins, label=r"$w$", alpha=0.7)
        ax.set(xlabel="Amount of workers", ylabel="Number of companies", title="Worker distribution at final time")
        ax.grid()
        
        # xticks
        number_of_xticks = 10
        xticks = np.linspace(0, max_val, number_of_xticks, dtype=int)
        ax.set_xticks(xticks)
        
        # Add parameters text
        if self.add_parameter_text_to_plot: self._add_parameters_text(ax)
        # Save and show
        self._save_fig(fig, "workforce")
        if self.show_plots: plt.show()


    def plot_system_money_mean_salary(self):
        """Plot system money spent and mean salary. Plot the fraction of companies bankrupt on top of the mean salary plot.
        """
        
        fig, (ax0, ax1) = plt.subplots(nrows=2)
        xlim = (0, self.time_steps)
        # ax0 - System money spent
        ax0.plot(self.time_values, self.system_money_spent)
        ax0.set(ylabel="Log $", title="System money spent", yscale="log", xlim=xlim)
        ax0.grid()
                
        # ax1 - Mean salary and bankruptcy fraction
        # Plot the bankruptcy fraction on a seperate axis
        ax1_2 = ax1.twinx()
        ax1_2.tick_params(axis='y', labelcolor="red")
        ax1_2.set_ylabel(ylabel="Bankruptcy fraction", color="red")
        ax1_2.plot(self.time_values, self.went_bankrupt / self.N, c="red", alpha=1)  # Bankruptcy fraction on twin axis
        
        # Mean salary
        ax1.plot(self.time_values, np.mean(self.s, axis=0), c="rebeccapurple", alpha=1)  # Mean salary
        ax1.axhline(y=self.ds, ls="--", c="grey", label=r"$\Delta s$")  # ds
        ax1.set(xlabel="Time", ylabel="Log Price", title="Mean salary and bankruptcy", xlim=xlim, yscale="log")
        ax1.grid()
        ax1.tick_params(axis='y', labelcolor="rebeccapurple")
        ax1.set_ylabel(ylabel="Mean Salary", color="rebeccapurple")        
        self._add_legend(ax1, y=0.9, ncols=1)
        
        # Add parameters text
        if self.add_parameter_text_to_plot: self._add_parameters_text(ax0)
        # Save and show
        self._save_fig(fig, "system_money")
        if self.show_plots: plt.show()


    def plot_debt(self):
        fig, (ax, ax1) = plt.subplots(nrows=2)
        d_diff = np.diff(self.d, axis=1)
        vmin = np.min((self.d.min(), d_diff.min()))
        vmax = np.max((self.d.max(), d_diff.max()))
        # Ax0 - Debt
        im = ax.imshow(self.d, cmap="coolwarm", norm=SymLogNorm(linthresh=1e-6, linscale=1e-6, vmin=vmin, vmax=vmax), aspect="auto")
        ax.set(xlabel="Time", ylabel="Company", title="Debt over time")
        fig.colorbar(im)
        
        # Ax1 - debt change
        im2 = ax1.imshow(d_diff, cmap="coolwarm", norm=SymLogNorm(linthresh=1e-6, linscale=1e-6, vmin=vmin, vmax=vmax), aspect="auto")
        ax1.set(xlabel="Time", ylabel="Company", title="Change in debt over time")        
        fig.colorbar(im2)
        
        # Add parameters text
        if self.add_parameter_text_to_plot: self._add_parameters_text(ax)
        # Save and show
        self._save_fig(fig, "debt")
        if self.show_plots: plt.show()
        

    def salary_analysis(self):
        """Plot salary of all companies over time, their difference from the mean salary, the mean salary and the fano factor.

        """
        # Calculations
        mean_salary = np.mean(self.s, axis=0)
        salary_diff = (self.s - mean_salary) / mean_salary
        salary_spread = np.std(self.s, axis=0) / mean_salary
        
        salary = self.s
        
        # Create figure
        xlim = (0, salary.shape[1])
        # fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4)
        fig, (ax0, ax1) = plt.subplots(nrows=2)

        # ax0 - mean salary and bankruptcy fraction
        color_salary = "rebeccapurple"
        ax0.plot(mean_salary, c=color_salary)
        # ax0.axhline(self.ds, ls="--", c="grey", label=r"Mutation magnitude")
        ax0.set(title="Mean salary and bankruptcy", yscale="log", xlim=xlim)
        ax0.tick_params(axis='y', labelcolor=color_salary)
        ax0.set_ylabel(ylabel="Log Mean Salary", color=color_salary)
        ax0.grid()
        
        ax0_2 = ax0.twinx()
        color_bankrupt = "darkred"
        ax0_2.plot(self.went_bankrupt / self.N, c=color_bankrupt, alpha=0.7)
        ax0_2.set_ylabel(ylabel="Bankruptcy", color=color_bankrupt)
        ax0_2.tick_params(axis='y', labelcolor=color_bankrupt)
        
        # ax1 - salary standard deviation
        ax1.plot(salary_spread)  
        ax1.set(xlabel="Time", ylabel="Log Price", title="Salary spread (std/mean)", yscale="log", xlim=xlim)      
        ax1.grid()
        
        # # ax2 - company salary values
        # im2 = ax2.imshow(salary, aspect="auto", norm=LogNorm(), cmap="magma")
        # ax2.set(ylabel="Companies", title="All company salaries", xlim=xlim)
        # fig.colorbar(im2, ax=ax2)        
        
        # # ax3 - salary percent difference from mean
        # im3 = ax3.imshow(salary_diff, aspect="auto", norm=SymLogNorm(linscale=1e-6, linthresh=1e-6), cmap="coolwarm")
        # ax3.set(ylabel="Companies", title="Salary percent difference from mean", xlim=xlim)
        # fig.colorbar(im3, ax=ax3)
        
        # Plot the peaks as vertical lines on ax0 and ax1
        if self.peak_idx is not None:
            for peak in self.peak_idx:
                ax0.axvline(x=peak, ls="--", c="grey", alpha=0.7)
                ax1.axvline(x=peak, ls="--", c="grey", alpha=0.7)
        
        # Add parameters text
        if self.add_parameter_text_to_plot: self._add_parameters_text(ax0)
        # Save and show
        self._save_fig(fig, "salary_analysis")
        if self.show_plots: plt.show()
        
    
    def s_min(self):
        # Load multiple files and get their salary values
        mean_salary_list = []
        salary_min_list = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]       
        peak_list = []
        period_mean = []
        period_std = []
        
        for salary_min in salary_min_list:
            with h5py.File(self.data_path, "r") as file:
                # Create the right group name
                parts = group_name.split("smin")
                new_group_name = parts[0] + f"smin{salary_min}"
                
                data_group = file[new_group_name]
                salary = data_group["s"][:]
                mean_salary = np.mean(salary, axis=0)
                mean_salary_list.append(mean_salary)
                
                # Find peak on mean salary
                peak_idx, _ = find_peaks(mean_salary, height=0.8e-2, prominence=0.5e-2, distance=50)
                peak_list.append(peak_idx[1:])  # First peak very likely to be in warmup
                period = np.diff(peak_idx)
                period_mean.append(np.mean(period))
                period_std.append(np.std(period))
        
        
        fig, (ax, ax1) = plt.subplots(nrows=2)
        c = ["rebeccapurple", "firebrick", "black", "darkorange", "seagreen", "deepskyblue"]
        def _plot_func(idx):
            # Mean salary
            ax.plot(mean_salary_list[idx], label=fr"$s_\text{{min}}=${salary_min_list[idx]}")
            ax.plot(peak_list[idx], mean_salary_list[idx][peak_list[idx]], "x", color="grey")
            ax.set(yscale="log", xlim=(0, self.time_steps))
            ax.grid()
        
            ax1.errorbar(salary_min_list[idx], period_mean[idx], yerr=period_std[idx], fmt="o", c=c[idx], )
        
        ax1.set(xlabel=r"$s_\text{min}$", ylabel="Peak period", title="Period of peaks", xscale="log")
        ax1.grid()
        
        for i in range(6):
            _plot_func(i)
            
        fig.suptitle(r"Mean salary for different $s_\text{min}$ values")
        
        #Legend
        self._add_legend(ax, x=0.5, y=0.75, ncols=3, fontsize=7)
        # Add parameters text
        if self.add_parameter_text_to_plot: self._add_parameters_text(ax)
        # Save and show
        self._save_fig(fig, "s_min")
        if self.show_plots: plt.show()


    def period_as_function_of_ds_and_rf(self):
        # Load data
        period_mean_arr = np.ones((len(ds_space), len(rf_space))) * 1e-8
        amplitude_mean_arr = np.ones((len(ds_space), len(rf_space))) * 1e-8
        
        for i, ds in enumerate(ds_space):
            self.ds = ds
            for j, rf in enumerate(rf_space):
                self.rf = rf
                with h5py.File(self.data_path, "r") as file:
                    gname = self._get_group_name()
                    data_group = file[gname]
                    period = data_group["peak_period"][:]
                    amplitude = data_group["peak_vals"][:]
                    if period.size == 0:
                        print(f"No peaks found for ds={ds}, rf={rf}")
                        continue
                    period_mean = np.mean(period)
                    amplitude_mean = np.mean(amplitude)
                    
                    period_mean_arr[i, j] = period_mean
                    amplitude_mean_arr[i, j] = amplitude_mean


        # Convert NaN to 1
        period_mean_arr = np.nan_to_num(period_mean_arr, nan=1e-8)
        amplitude_mean_arr = np.nan_to_num(amplitude_mean_arr, nan=1e-8)
        
        # Create figure
        fig, ax = plt.subplots(ncols=1, nrows=2)
        
        ax_period = ax[0]
        ax_amplitude = ax[1]

        # Contour plotted
        # vmin_period = period_mean.min()+ 1e-10
        vmin_period = 1
        vmax_period = period_mean_arr.max()
        vmin_amplitude = amplitude_mean_arr.min() + 1e-10
        vmax_amplitude = amplitude_mean_arr.max()

        # Cmap
        # cmap_period = plt.cm.get_cmap("viridis")
        cmap_period = plt.colormaps.get_cmap("hot")
        cmap_amplitude = plt.colormaps.get_cmap("hot")
        
        # Contourf
        # ax_period.contourf(ds_space, rf_space, period_mean_arr, cmap="magma", vmin=vmin_period, vmax=vmax_period)
        # ax_amplitude.contourf(ds_space, rf_space, amplitude_mean_arr, cmap="hot", vmin=vmin_amplitude, vmax=vmax_amplitude)
        
        # Imshow        
        ax_period.imshow(period_mean_arr, aspect="auto", cmap=cmap_period, vmin=vmin_period, vmax=vmax_period, extent=[ds_space.min(), ds_space.max(), rf_space.min(), rf_space.max()], origin="lower")
        ax_amplitude.imshow(amplitude_mean_arr, aspect="auto", cmap=cmap_amplitude, vmin=vmin_amplitude, vmax=vmax_amplitude, extent=[ds_space.min(), ds_space.max(), rf_space.min(), rf_space.max()], origin="lower")
        
        # Add colorbars
        sm_period = plt.cm.ScalarMappable(cmap=cmap_period, norm=LogNorm(vmin=vmin_period, vmax=vmax_period))
        fig.colorbar(sm_period, ax=ax_period, orientation="vertical", label="Period", )
        sm_amplitude = plt.cm.ScalarMappable(cmap=cmap_amplitude, norm=LogNorm(vmin=vmin_amplitude, vmax=vmax_amplitude))
        fig.colorbar(sm_amplitude, ax=ax_amplitude, orientation="vertical", label="Amplitude", )            
        
        # Labels
        ax_period.set(title="Period mean", ylabel=r"$\Delta s$", xlabel=r"$rf$")
        ax_amplitude.set(title="Amplitude mean", ylabel=r"$\Delta s$", xlabel=r"$rf$")
        
        # Save and show
        self._save_fig(fig, "period_ds_rf")
        if self.show_plots: plt.show()


    def time_scale(self):
        """Investigate how different options for the time scale affect the period of the peaks.
        Plot:
            1. Mean salary over time with its peaks
            2. Period of the peaks as a function of the time scale
        """
        # Load data
        with h5py.File(self.data_path, "r") as file:
            # Define group names
            group_0 = self.group_name + "_time_scale_0"
            group_x = self.group_name + "_time_scale_x"
            group_inverse_r = self.group_name + "_time_scale_inverse_r"
            
            # Get data
            data_group1 = file[group_0]
            data_group2 = file[group_x]
            data_group3 = file[group_inverse_r]
            
            # Get salary values
            salary_0 = np.mean(data_group1["s"][:], axis=0)
            salary_x = np.mean(data_group2["s"][:], axis=0)
            salary_inverse_r = np.mean(data_group3["s"][:], axis=0)
            
            self.time_steps = salary_0.size
        
        # Calculate peaks for each time scale
        def _get_peaks(salary):
            peak_idx, _ = find_peaks(salary, height=0.8e-2, prominence=0.5e-2, distance=300, width=5)
            peak_idx = peak_idx[1:]  # First peak very likely to be in warmup
            period = np.diff(peak_idx)
            period_mean = np.mean(period)
            period_std = np.std(period)
            return peak_idx, period_mean, period_std
        
        peak_idx_0, period_0, period_std_0 = _get_peaks(salary_0)
        peak_idx_x, period_x, period_std_x = _get_peaks(salary_x)
        peak_idx_inverse_r, period_inverse_r, period_std_inverse_r = _get_peaks(salary_inverse_r)
        
        # Create figure
        fig, (ax_s, ax_T) = plt.subplots(nrows=2)
        
        # ax_s - Salary
        ax_s.plot(salary_0, label="Time scale 0")
        ax_s.plot(salary_x, label="Time scale x")
        ax_s.plot(salary_inverse_r, label="Time scale 1/r")        
        ax_s.plot(peak_idx_0, salary_0[peak_idx_0], "x", c="grey")
        ax_s.plot(peak_idx_x, salary_x[peak_idx_x], "x", c="grey")
        ax_s.plot(peak_idx_inverse_r, salary_inverse_r[peak_idx_inverse_r], "x", c="grey")

        ax_s.set(yscale="log", xlim=(0, self.time_steps), xlabel="Time", ylabel="Mean Salary")
        ax_s.grid()
        self._add_legend(ax_s, ncols=3)
        
        # ax_T - Period
        for i in range(3):
            ax_T.errorbar([0, 1, 2][i], [period_0, period_x, period_inverse_r][i], yerr=[period_std_0, period_std_x, period_std_inverse_r][i], fmt="o")
        ax_T.set(ylabel="Period", title="Period of peaks")
        ax_T.grid()
        ax_T.set_xticks([0, 1, 2], labels=["T=0", "T=50", "T=1/r"])
        
        # Add parameters
        if self.add_parameter_text_to_plot: self._add_parameters_text(ax_s)
        # Save and show
        self._save_fig(fig, "time_scale")
        if self.show_plots: plt.show()
        

    # -- Peak analysis --
    def _load_peak_data(self):
      # Empty lists
        self.system_money_spent_list = []
        self.peak_list = []
        self.peak_idx_list = []

        def _replace_salary_increase(group_name, new_rho_val):
            parts = group_name.split("ds")
            new_group_name = parts[0] + f"ds{new_rho_val}"
            return new_group_name
        
        with h5py.File(self.data_path, "r") as file:
            self.parameter_space = ds_space
            for rho in ds_space:
                # Find "rho" in group_name, replace the value with the current rho
                group_name_new = _replace_salary_increase(self.group_name, rho)
                data_group = file[group_name_new]

                system_money_spent = data_group["system_money_spent"][:]
                peak_idx = data_group["peak_idx"][:]
                peak_vals = data_group["peak_vals"][:]
                
                # Append to lists
                self.system_money_spent_list.append(system_money_spent)
                self.peak_list.append(peak_vals)
                self.peak_idx_list.append(peak_idx)
            

    def parameter_peak(self):
        self._load_peak_data()
        
        number_of_salary_values = len(self.peak_idx_list)
        ncols = 2
        nrows = number_of_salary_values // ncols
        
        def _plot_func(axis, idx):
            axis.plot(self.time_values, self.system_money_spent_list[idx])
            # axis.plot(self.peak_idx_list[idx], self.peak_list[idx], "x")
            axis.set(ylabel="Log Price", title=fr"$\rho=${self.parameter_space[idx]}", yscale="log", xlabel="Time", xlim=(0, self.time_steps))
            axis.grid()
        
        fig, ax = plt.subplots(ncols=ncols, nrows=nrows)
        # Transform ax to a 2d array if it is not already
        if nrows == 1: ax = ax.reshape(1, -1)
        
        for i in range(ncols*nrows):
            _plot_func(ax[i // ncols, i % ncols], i)
        
        fig.suptitle(r"System money spent for different $\Delta s$ values")
        
        # Add parameters text
        if self.add_parameter_text_to_plot: self._add_parameters_text(ax[0, 0])
        # Save and show
        self._save_fig(fig, "peak_system_money")
        if self.show_plots: plt.show()
        
    
    def peak_frequency_vs_parameter(self):
        self._load_peak_data()
        
        period_mean = np.zeros(len(self.peak_idx_list))
        period_std = np.zeros(len(self.peak_idx_list))
        for i in range(len(self.peak_idx_list)):
            if self.peak_idx_list[i].size == 0:
                period_mean[i] = 0
                period_std[i] = 0
            elif self.peak_idx_list[i].size == 1:
                period_mean[i] = self.peak_idx_list[i][0]
                period_std[i] = 0
            else:
                peak_diff = np.diff(self.peak_idx_list[i])  # Distances between neighbouring peaks
                period_mean[i] = np.mean(peak_diff)  # Mean distance
                period_std[i] = np.std(peak_diff)  # Standard deviation of distances
        
        # freq_mean = 1 / period_mean
        # freq_std = period_std / period_mean ** 2  # Error propagation: std(f) = 1 / T * std(T) / T

        # Create figure
        fig, ax = plt.subplots()
        ax.errorbar(self.parameter_space, period_mean, yerr=period_std, fmt="o")
        ax.set(xlabel=r"$\rho$", ylabel="Peak period", title="Period of peaks")
        ax.grid()
        
        # Add parameters text
        if self.add_parameter_text_to_plot: self._add_parameters_text(ax)
        # Save and show
        self._save_fig(fig, "peak_period")
        if self.show_plots: plt.show()
    
    
    def peak_first_crash(self):
        # Get the time of the first peak which corresponds to the time it takes for the system to crash
        self._load_peak_data()
        
        first_crash_time = np.zeros(len(self.peak_idx_list))
        
        for i in range(len(ds_space)):
            # Check for the case of no peaks
            if np.size(self.peak_idx_list) == 0:
                first_crash_time[i] = self.time_steps
            else:
                first_crash_time[i] = self.peak_idx_list[i][0]
                
        # Find ylim as the maximum value NOT equal to time_steps
        ymax = np.max(first_crash_time[first_crash_time != self.time_steps]) * 1.1  # 10% bigger
        ylim = (0, ymax)    
        # Find the first rho value for which the companies start having no distinct peaks i.e. the first time value to be equal to time_steps
        no_distinct_peaks = self.parameter_space[first_crash_time == self.time_steps][0]
        
        # Create figure
        fig, ax = plt.subplots()
        ax.plot(self.parameter_space, first_crash_time, "o", label="Time of first crash")
        ax.axvline(x=no_distinct_peaks, ls="--", c="grey", label="No distinct peaks")
        ax.set(xlabel=r"$\rho$", ylabel="Time", ylim=ylim, title="Time of first crash")
        ax.grid()
        
        # Legend
        self._add_legend(ax, y=0.9, ncols=2)
        # Add parameters text
        if self.add_parameter_text_to_plot: self._add_parameters_text(ax)
        # Save and show
        self._save_fig(fig, "first_crash_time")
        if self.show_plots: plt.show()
    

    def animate_size_distribution(self):
        self.w = self.w[:, 4000:]  # Remove first 4000 time steps
        
        # Bin data
        bins = np.arange(0, int(self.w.max()) + 1, 1)
        # Figure setup        
        fig, ax = plt.subplots()
        _, _, bar_container = ax.hist(self.w[:, 0], bins)  # Initial histogram 
        ax.set(xlim=(bins[0], bins[-1]), title="Time = 0", xlabel="Number of employees", ylabel="Counts")
        ax.grid()
        xticks = np.linspace(0, bins[-1], 10, dtype=int)
        ax.set_xticks(xticks)
        
        # Text
        if self.add_parameter_text_to_plot: self._add_parameters_text(ax)

        def animate(i, bar_container):
            """Frame animation function for creating a histogram."""
            # Histogram
            data = self.w[:, i]
            n, _ = np.histogram(data, bins)
            for count, rect in zip(n, bar_container.patches):
                rect.set_height(count)
            
            # Title
            ax.set_title(f"Time = {i}")
            return bar_container.patches
        
        # Create the animation
        anim = functools.partial(animate, bar_container=bar_container)  # Necessary when making histogram
        ani = animation.FuncAnimation(fig, anim, frames=self.time_steps, interval=1)
        
        # Save animation
        self._save_anim(ani, name="workforce_animation")
  
  
if __name__ == "__main__":
    show_plots = True
    add_parameter_text_to_plot = True
    bank_vis = BankVisualization(group_name, show_plots, add_parameter_text_to_plot)
    
    
    print("Started plotting")
    plot_company = True
    if plot_company:
        # bank_vis.plot_companies(4)
        # bank_vis.plot_means()
        # bank_vis.plot_interest_rates()
        # bank_vis.plot_system_money_mean_salary()
        # bank_vis.plot_production_capacity()
        # bank_vis.salary_analysis()
        # bank_vis.plot_debt()
        
        # bank_vis.s_min()
        bank_vis.period_as_function_of_ds_and_rf()
        # bank_vis.time_scale()
        
    # -- Peak analysis -- 
    plot_peak = False
    if plot_peak:
        bank_vis.parameter_peak()
        # bank_vis.peak_frequency_vs_parameter()
        
        # bank_vis.peak_first_crash()  # Basically identical to frequency 
    
    # -- Animations --    
    # bank_vis.animate_size_distribution()
    
    print("Finished plotting")