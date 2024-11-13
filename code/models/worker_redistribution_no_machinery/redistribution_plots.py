import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from pathlib import Path
import h5py
import functools
import matplotlib.animation as animation

# My files
import general_functions
from redistribution_no_m_master import dir_path_output, dir_path_image, group_name


class BankVisualization(general_functions.PlotMethods):
    def __init__(self, group_name, show_plots, add_parameter_text_to_plot):
        super().__init__(group_name)
        self.group_name = group_name
        self.show_plots = show_plots
        self.add_parameter_text_to_plot = add_parameter_text_to_plot
        
        # Check if the path to the image folder exists, otherwise create it
        dir_path_image.mkdir(parents=True, exist_ok=True)
        self.dir_path_image = dir_path_image
        self.data_path = dir_path_output / "redistribution_no_m_simulation_data.h5"
        
        # Load data        
        with h5py.File(self.data_path, "r") as file:
            data_group = file[group_name]
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

    
    def plot_companies(self, N_plot):
        w_plot = self.w[: N_plot, :].T
        d_plot = self.d[: N_plot, :].T
        
        fig, (ax, ax1) = plt.subplots(nrows=2)

        ax.plot(self.time_values, w_plot)
        ax.set(ylabel="Number of workers", title="Workforce")
        ax.grid()

        ax1.plot(self.time_values, d_plot)
        ax1.set(ylabel="Price", title="Debt", xlabel="Time")
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
        ax.set(xlabel="Time", ylabel="Log Price", title="Mean values", yscale="log")
        ax.grid()
        self._add_legend(ax, ncols=4, y=0.9)
        
        # Display parameters
        if self.add_parameter_text_to_plot: self._add_parameters_text(ax)
        # Save and show
        self._save_fig(fig, "means")
        if self.show_plots: plt.show()
                
        
    def plot_interest_rates_unemployed(self):
        """Plot the interest rate and the fraction of companies gone bankrupt over time.
        """
        fig, (ax, ax1, ax2) = plt.subplots(nrows=3)

        # ax Interest rate and free interest rate
        ax.axhline(y=0.05, ls="--", label="Interest rate free")
        ax.plot(self.time_values, self.interest_rate, label="Interest rate", c="firebrick")
        ax.set(ylabel="Interest rate")
        ax.grid()
        ax.legend(bbox_to_anchor=(0.5, 1), loc="lower center", fontsize=10, ncols=3)
        
        # ax1 bankruptcies
        ax1.plot(self.time_values, self.went_bankrupt / self.N, label="Bankruptcies")
        ax1.set(ylabel="Bankrupt fraction", title="Bankruptcies")
        ax1.grid()
        
        # ax2 unemploed
        ax2.plot(self.time_values, self.unemployed / self.W)
        ax2.set(xlabel="Time", ylabel="Fraction unemployed", title="Fraction of workforce unemployed")
        ax2.grid()
        
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
        
        
    def plot_salary(self):
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12, 10))
        
        # All companies' salary over time
        # im = ax.imshow(self.s, cmap="magma")
        # ax.set(xlabel="Time", ylabel="Company", title="Salary over time")
        # fig.colorbar(im)
        
        # ax1 - Mean salary over time
        ax1.plot(self.time_values, np.mean(self.s, axis=0))
        ax1.set(xlabel="Time", ylabel="Log Price", title="Mean salary over time", xlim=(0, self.time_steps), yscale="log")
        ax1.grid()
        
        # ax2 - Delta d over time
        delta_d = np.diff(self.d, axis=1)
        im2 = ax2.imshow(delta_d, cmap="coolwarm", norm=SymLogNorm(linthresh=1e-6, linscale=1e-6))
        ax2.set(xlabel="Time", ylabel="Company", title="Delta debt over time")
        fig.colorbar(im2)
        
        # Add parameter text
        if self.add_parameter_text_to_plot: self._add_parameters_text(ax1)
        # Save and show
        self._save_fig(fig, "salary")
        if self.show_plots: plt.show()
        
        
    def plot_production_capacity(self):
        w_final = self.w[:, -1]
        max_val = np.max(w_final)
        bins = np.arange(0, max_val + 1 , 1)
        
        fig, ax  = plt.subplots()
        counts, edges, _ = ax.hist(w_final, bins=bins, label=r"$w$", alpha=0.7)
        ax.set(xlabel="Amount of workers", ylabel="Number of companies", title="Worker and machine distributions at final time")
        ax.grid()
        
        # xticks
        number_of_xticks = 10
        xticks = np.linspace(0, max_val, number_of_xticks, dtype=int)
        ax.set_xticks(xticks)
        
        # Legend
        self._add_legend(ax, ncols=2, y=0.9)
        
        # Add parameters text
        if self.add_parameter_text_to_plot: self._add_parameters_text(ax)
        # Save and show
        self._save_fig(fig, "workforce")
        if self.show_plots: plt.show()


    def plot_unemployed(self):
        fig, ax = plt.subplots()
        ax.plot(self.time_values, self.unemployed / self.W)
        ax.set(xlabel="Time", ylabel="Fraction", title="Fraction of workforce unemployed")
        ax.grid()
        
        # Add parameters text
        if self.add_parameter_text_to_plot: self._add_parameters_text(ax)
        # Save and show
        self._save_fig(fig, "unemployed")
        if self.show_plots: plt.show()


    def plot_system_money(self):
        """Plot system money spent, fraction employed and inflation
        """
        
        fig, ax0 = plt.subplots(nrows=1)
        
        # ax0 - System money spent
        ax0.plot(self.time_values, self.system_money_spent)
        ax0.set(ylabel="Log $", title="System money spent", yscale="log")
        ax0.grid()
                
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
        

    def _load_peak_data(self):
      # Empty lists
        self.system_money_spent_list = []
        self.peak_list = []
        self.peak_idx_list = []

        def _replace_salary_increase(group_name, new_rho_val):
            parts = group_name.split("rho")
            new_group_name = parts[0] + f"rho{new_rho_val}"
            return new_group_name
        
        
        with h5py.File(self.data_path, "r") as file:
            self.parameter_space = self.salary_increase_space
            for rho in self.salary_increase_space:
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
            if (self.system_money_spent_list[idx] <= 0).any():
                print("Contained zero or negative values")
            
            axis.plot(self.time_values, self.system_money_spent_list[idx])
            axis.plot(self.peak_idx_list[idx], self.peak_list[idx], "x")
            axis.set(ylabel="Log System money spent", title=fr"$\rho=${self.parameter_space[idx]}", yscale="log", xlabel="Time")
            axis.grid()
        
        fig, ax = plt.subplots(ncols=ncols, nrows=nrows)
        for i in range(ncols*nrows):
            _plot_func(ax[i // ncols, i % ncols], i)
        
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
        
        for i in range(len(self.salary_increase_space)):
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
        # Bin data
        bins = np.arange(0, int(self.p.max()) + 1, 1)
        # Figure setup        
        fig, ax = plt.subplots()
        _, _, bar_container = ax.hist(self.p[:, 0], bins)  # Initial histogram 
        ax.set(xlim=(bins[0], bins[-1]), title="Time = 0", xlabel="Number of employees", ylabel="Counts")
        ax.grid()
        xticks = np.linspace(0, bins[-1], 10, dtype=int)
        ax.set_xticks(xticks)
        
        # Text
        if self.add_parameter_text_to_plot: self._add_parameters_text(ax)

        def animate(i, bar_container):
            """Frame animation function for creating a histogram."""
            # Histogram
            data = self.p[:, i]
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
        # bank_vis.plot_interest_rates_unemployed()
        # bank_vis.plot_system_money()
        # bank_vis.plot_production_capacity()
        # bank_vis.plot_salary()
        bank_vis.plot_debt()
        
    # -- Peak analysis -- 
    plot_peak = False
    if plot_peak:
        bank_vis.parameter_peak()
        bank_vis.peak_frequency_vs_parameter()
        
        # bank_vis.peak_first_crash()  # Basically identical to frequency 
    
    # -- Animations --    
    # bank_vis.animate_size_distribution()
    
    print("Finished plotting")