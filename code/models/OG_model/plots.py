import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from pathlib import Path
import h5py
import functools
import matplotlib.animation as animation

# My files
import general_functions
from master import dir_path_output, dir_path_image, group_name


class Visualization(general_functions.PlotMethods):
    def __init__(self, file_group_name, show_plots, add_parameter_text_to_plot):
        super().__init__(file_group_name)
        self.group_name = file_group_name
        self.show_plots = show_plots
        self.add_parameter_text_to_plot = add_parameter_text_to_plot
        
        
        # Check if the path to the image folder exists, otherwise create it
        dir_path_image.mkdir(parents=True, exist_ok=True)
        self.dir_path_image = dir_path_image
        self.data_path = dir_path_output / "DebtDeflation.h5"
        
        # Load data        
        with h5py.File(self.data_path, "r") as file:
            data_group = file[group_name]
            # Print the names of all groups in file
            self.filename = data_group.name.split("/")[-1]
            
            # Company
            self.d = data_group["d"][:]
            self.p = data_group["p"][:]
            self.m = data_group["m"][:]
            
            # System
            self.interest_rate = data_group["interest_rate"][:]
            self.interest_rate_free = data_group["interest_rate_free"][:]
            self.went_bankrupt = data_group["went_bankrupt"][:]
            
        self.N = self.p.shape[0]
        self.time_steps = self.p.shape[1]
        self.time_values = np.arange(self.time_steps)
        
        
    def plot_companies(self, N_plot):
        p_plot = self.p[: N_plot, :].T
        m_plot = self.m[: N_plot, :].T
        d_plot = self.d[: N_plot, :].T
        
        fig, (ax, ax1, ax2) = plt.subplots(nrows=3)

        ax.plot(self.time_values, p_plot)
        ax1.plot(self.time_values, d_plot)
        ax2.plot(self.time_values, m_plot)
        
        ax.set(ylabel="Log Price", title="Production", yscale="log")
        ax1.set(ylabel="Log Price", title="Debt", yscale="log")
        ax2.set(ylabel="Log Price", title="Money", xlabel="Time", yscale="log")
        
        ax.grid()
        ax1.grid()
        ax2.grid()

        # Display parameter values
        if self.add_parameter_text_to_plot: self._add_parameters_text(ax)
        
        # Save figure
        self._save_fig(fig, "singlecompanies")
        if self.show_plots: plt.show()
        
        
    def plot_means(self):
        d_mean = np.mean(self.d, axis=0)
        p_mean = np.mean(self.p, axis=0)
        m_mean = np.mean(self.m, axis=0)
        
        fig, ax = plt.subplots()
        ax.plot(self.time_values, p_mean, label=r"$p$")
        ax.plot(self.time_values, d_mean, label=r"$d$")
        ax.plot(self.time_values, m_mean, label=r"$m$")
        
        ax.set(ylabel="Log Price", title="Company mean", xlabel="Time", yscale="log")
        ax.grid()
        
        self._add_legend(ax, ncols=3)
        # Parameter text
        if self.add_parameter_text_to_plot: self._add_parameters_text(ax)
        # Save and show
        self._save_fig(fig, "company_mean")
        if self.show_plots: plt.show()
        
        
    def plot_interest_rate_and_bankruptcy(self):
        
        fig, (ax0, ax1) = plt.subplots(nrows=2)
        
        # ax0 interest rate
        ax0.plot(self.time_values, self.interest_rate, "--", label="Interest rate")
        ax0.plot(self.time_values, self.interest_rate_free, label="Free interest rate")
        ax0.set(ylabel="Interest rate", title="Interest rate")
        self._add_legend(ax0, ncols=2)
        
        # ax1 bankruptcy
        ax1.plot(self.time_values, self.went_bankrupt / self.N)
        ax1.set(ylabel="Fraction", title="Bankrupt", xlabel="Time")
        
        ax0.grid()
        ax1.grid()
        
        # Display parameter values
        if self.add_parameter_text_to_plot: self._add_parameters_text(ax0)
        # save and show
        self._save_fig(fig, "interest_rate_bankruptcy")
        if self.show_plots: plt.show()
        

if __name__ == "__main__":
    show_plots = True
    add_parameter_text_to_plot = True
    vis = Visualization(group_name, show_plots, add_parameter_text_to_plot)
    
    print("Started plotting")
    
    # Plot companies
    vis.plot_companies(N_plot=4)
    vis.plot_means()
    vis.plot_interest_rate_and_bankruptcy()