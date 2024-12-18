import general_functions 
import matplotlib.pyplot as plt
import numpy as np
import h5py
from pathlib import Path
from run import file_path, group_name, dir_path_image


class PlotMaster(general_functions.PlotMethods):
    def __init__(self, data_group_name, show_plots=True, add_parameter_text_to_plot=True, save_figure=True):
        super().__init__(data_group_name)
        # Allow for the option of specifying a group name. Otherwise the existing group name is used.
        self.group_name = data_group_name
        self.show_plots = show_plots
        self.save_figure = save_figure
        self.add_parameter_text_to_plot = add_parameter_text_to_plot
        self.dir_path_image = dir_path_image
        self.loaded_groups = {}
        
        self.skip_values = 5000
        
        
    def _load_data_group(self, gname):
        with h5py.File(file_path, "r") as f:
            group = f[gname]
            data = {
                "w": group["w"][:],
                "d": group["d"][:],
                "s": group["s"][:],
                "r": group["r"][:],
                "went_bankrupt": group["went_bankrupt"][:],
                "mu": group["mu"][:],
                "mutations": group["mutations"][:],
                "N": group.attrs["N"],
                "W": group.attrs["W"],
                "ds": group.attrs["ds"],
                "rf": group.attrs["rf"],
                "m": group.attrs["m"],
            }
            try:
                data["peak_idx"] = group["peak_idx"][:]
            except KeyError:
                data["peak_idx"] = None
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
        self.mu = data["mu"]
        self.mutations = data["mutations"]
        self.N = data["N"]
        self.W = data["W"]
        self.ds = data["ds"]
        self.rf = data["rf"]
        self.m = data["m"]
        self.peak_idx = data["peak_idx"]
        self.time_steps = self.s.shape[1]
        self.xlim = (self.skip_values, self.time_steps)
        
        if self.time_steps <= self.skip_values:
            self.skip_values = 0
            print(f"Skip values {self.skip_values} is larger than the time steps {self.time_steps}. Set skip values to 0.")
    
    
    def plot_salary(self):
        """Plot the mean salary and fraction who went bankrupt on twinx. Plot the spread (std/mean) on a subplot below it."""
        self._get_data(self.group_name)
        mean_salary = self.s.mean(axis=0)[self.skip_values:]
        median_salary = np.median(self.s, axis=0)[self.skip_values:]
        fraction_bankrupt = (self.went_bankrupt[self.skip_values:] / self.N)
        spread = (self.s.std(axis=0)[self.skip_values:] / mean_salary)
        time_values = np.arange(self.skip_values, self.time_steps)
        
        # Create figure
        fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(10, 10))
        
        # ax0 - Salary and fraction who went bankrupt
        c0 = general_functions.list_of_colors[0]
        c1 = general_functions.list_of_colors[1]
        
        ax0.plot(time_values, mean_salary, label="Mean salary", c=c0, alpha=1)
        ax0.plot(time_values, median_salary, label="Median salary", c="black", alpha=1, ls="dotted")
        
        ax0.set(xlim=self.xlim, ylabel="Log Price", yscale="log", title="Mean salary and bankruptcies")
        ax0.set_ylabel("Log Price", color=c0)
        ax0.tick_params(axis='y', labelcolor=c0)
        ax0.grid()
        self._add_legend(ax0, ncols=2, x=0.5, y=0.9)

        ax0_twin = ax0.twinx()
        ax0_twin.plot(time_values, fraction_bankrupt, color=c1, label="Fraction bankrupt", alpha=0.6)
        ax0_twin.set_ylabel("Fraction bankrupt", color=c1)
        ax0_twin.tick_params(axis='y', labelcolor=c1)
        
        # ax1 - Spread
        ax1.plot(time_values, spread, label="Spread")
        ax1.set(xlabel="Time", xlim=self.xlim, ylabel="Spread", title="Spread (std/mean)")
        ax1.grid()
        
        # Plot the peaks as vertical lines on ax0 and ax1
        if self.peak_idx is not None:
            for peak in self.peak_idx:
                ax0.axvline(x=peak, ls="--", c="grey", alpha=0.7)
                ax1.axvline(x=peak, ls="--", c="grey", alpha=0.7)
        
        self._text_save_show(fig, ax0, "salary", xtext=0.05, ytext=0.75, fontsize=6)
        
        
    def plot_single_companies(self, N_plot):
        """Plot the salary and debt of the first N_plot companies"""
        # Get data and remove skip_values
        self._get_data(self.group_name)
        
        # Get first N_plot companies
        s = self.s[:N_plot, self.skip_values:]
        d = self.d[:N_plot, self.skip_values:]
        w = self.w[:N_plot, self.skip_values:]
        time_values = np.arange(self.skip_values, self.time_steps)
        
        # Create figure
        fig, (ax_s, ax_d, ax_w) = plt.subplots(nrows=3)
        
        # c0 = general_functions.list_of_colors[0]
        # c1 = general_functions.list_of_colors[1]
        # ax_s - salary
        ax_s.plot(time_values, s.T)
        ax_s.set(title=f"Salary and debt of first {N_plot} companies", yscale="log")
        ax_s.set_ylabel("Log salary")
        # ax_s.tick_params(axis='y', labelcolor=c0)
        ax_s.grid()
        
        # ax_d - debt
        # ax_d = ax_s.twinx()
        ax_d.plot(time_values, d.T,)
        ax_d.set(xlabel="Time", ylabel="Log Debt", yscale="symlog")
        ax_d.set_ylabel("Debt")
        # ax_d.tick_params(axis='y', labelcolor=c1)
        ax_d.grid()
        
        # ax_w - workers
        ax_w.plot(time_values, w.T)
        ax_w.set(xlabel="Time", ylabel="Workers", title="Workers")
        
        self._text_save_show(fig, ax_s, "single_companies", xtext=0.05, ytext=0.85)
        
        
    def plot_debt(self):
        """Plot the mean debt and fraction who went bankrupt on twinx and below it debt together with salary, last subplot has debt distribution at final time step. 
        """
        # Preprocess
        self._get_data(self.group_name)
        mean_debt = self.d.mean(axis=0)[self.skip_values:]
        median_debt = np.median(self.d, axis=0)[self.skip_values:]
        mean_salary = self.s.mean(axis=0)[self.skip_values:]
        fraction_bankrupt = (self.went_bankrupt[self.skip_values:] / self.N)
        time_values = np.arange(self.skip_values, self.time_steps)
        d_final = self.d[:, -1]
        
        # Create figure
        fig, (ax, ax1, ax2) = plt.subplots(nrows=3)
        c0 = general_functions.list_of_colors[0]
        c1 = general_functions.list_of_colors[1]
        
        ax.plot(time_values, mean_debt, c=c0, label="Mean debt")
        # ax.plot(time_values, median_debt, c=c0, ls="--", label="Median debt")
        ax.set(xlabel="Time", title="Mean debt and bankruptcies", yscale="symlog")
        ax.set_ylabel("Log Price", color=c0)
        ax.tick_params(axis='y', labelcolor=c0)
        ax.grid()
        self._add_legend(ax, ncols=2, x=0.7, y=0.7, fontsize=6)
        
        ax_twin = ax.twinx()
        ax_twin.plot(time_values, fraction_bankrupt, color=c1, label="Fraction bankrupt", alpha=0.6)
        ax_twin.set_ylabel("Fraction bankrupt", color=c1)
        ax_twin.tick_params(axis='y', labelcolor=c1)
        
        # ax1 - Salary and debt
        c2 = general_functions.list_of_colors[2]
        ax1.plot(time_values, mean_debt, c=c0)
        ax1.set(xlabel="Time", title="Mean salary and debt", yscale="symlog")
        ax1.set_ylabel("Mean Debt", color=c0)
        ax1.tick_params(axis='y', labelcolor=c0)
        ax1.grid()
        
        ax1_twin = ax1.twinx()
        ax1_twin.plot(time_values, mean_salary, c=c2, alpha=0.7)
        ax1_twin.set_ylabel("Log mean salary", color=c2)
        ax1_twin.tick_params(axis='y', labelcolor=c2)
        ax1_twin.set_yscale("log")
        
        # ax2 - Debt distribution
        Nbins = int(np.sqrt(self.N))
        ax2.hist(d_final, bins=Nbins)
        ax2.set(title="Debt distribution at final time step", xlabel="Debt", ylabel="Counts", yscale="log")
        ax2.grid()
        
        # Log scale hist requires only positive values
        # self._xlog_hist(d_final, fig, ax2, xlabel="Log Debt", ylabel="Counts", title="Debt distribution at final time step")
        
        self._text_save_show(fig, ax, "debt", xtext=0.05, ytext=0.85)
        
        
    def plot_collective(self):
        """Plot the mean salary and fraction bankrupt on a shared x axis, plot debt in another subplot below it, plot interest rate in a third subplot.
        """
        # Preprocess
        self._get_data(self.group_name)
        s_mean = self.s.mean(axis=0)[self.skip_values:]
        d_mean = self.d.mean(axis=0)[self.skip_values:]
        w_mean = self.w.mean(axis=0)[self.skip_values:]
        w_bankrupt = self.went_bankrupt[self.skip_values:]
        r = self.r[self.skip_values:]
        time_values = np.arange(self.skip_values, self.time_steps)
        
        fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(14, 10))
        ax_s = ax[0, 0]
        ax_d = ax[1, 0]
        ax_r = ax[0, 1]
        ax_w = ax[1, 1]
        
        # ax_s - Salary and bankrupt
        c0 = general_functions.list_of_colors[0]
        c1 = general_functions.list_of_colors[1]
        ax_s_twin = ax_s.twinx()
        ax_s_twin.plot(time_values, w_bankrupt / self.N, c=c1, alpha=0.7)
        ax_s_twin.set_ylabel("Fraction bankrupt", color=c1)
        ax_s_twin.tick_params(axis='y', labelcolor=c1)

        ax_s.plot(time_values, s_mean, c=c0)
        ax_s.tick_params(axis='y', labelcolor=c0)
        ax_s.set_ylabel("Log salary", color=c0)
        ax_s.set(title="Mean salary and bankruptcies", yscale="log")
        ax_s.grid()

        # ax_d - Debt
        c2 = general_functions.list_of_colors[2]
        ax_d.plot(time_values, d_mean, c=c2)
        ax_d.set(title="Log Mean debt", xlabel="Time", yscale="symlog")
        ax_d.grid()

        # ax_r - Interest rate
        c3 = general_functions.list_of_colors[3]
        ax_r.plot(time_values, r, c=c3)
        ax_r.set(title="Interest rate", xlabel="Time")
        ax_r.grid()
        
        # ax_w - Workers
        c4 = general_functions.list_of_colors[4]
        ax_w.plot(time_values, w_mean, c=c4)
        ax_w.set(title="Workers", xlabel="Time")
        ax_w.grid()

        self._text_save_show(fig, ax_s, "collective", xtext=0.05, ytext=0.85)
        
    
    def plot_mutations(self):
        """Compare mean salary to the sum of mutations
        """
        # Preprocess
        self._get_data(self.group_name)
        s_mean = self.s.mean(axis=0)
        
        fig, ax = plt.subplots()
        c0 = general_functions.list_of_colors[0]
        c1 = general_functions.list_of_colors[1]
        ax.plot(self.mutations, label="Sum of mutations", ls="--", alpha=0.6, c=c1)
        ax.plot(s_mean, label="Mean salary", alpha=0.95, c=c0)
        ax.set(title="Mean salary and sum of mutations", xlabel="Time", ylabel="Value", yscale="symlog")
        ax.grid()
        
        self._add_legend(ax, ncols=2)
        
        self._text_save_show(fig, ax, "mutations", xtext=0.05, ytext=0.85)