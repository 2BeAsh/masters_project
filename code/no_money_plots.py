import numpy as np
from time import time
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.animation as animation
from pathlib import Path
from tqdm import tqdm
import h5py

# My files
import general_functions  # Own library, here only used for matplotlib styling
from debt_deflation_plots import DebtDeflationVisualization
from no_money_master import first_group_params, dir_path_output, dir_path_image


class BankVisualization(DebtDeflationVisualization):
    def __init__(self, group_name, show_plots):
        super().__init__(group_name, show_plots)
        
        
        # Variable instances
        self.group_name = group_name
        self.show_plots = show_plots
        
        self.dir_path_image = dir_path_image
        
        # Load the simulation_data.h5 file using h5py
        self.data_path = dir_path_output / 'no_money_data.h5'
        
        with h5py.File(self.data_path, "r") as file:
            data_group = file[group_name]
            self.filename = data_group.name.split("/")[-1]  # Remove the "/" at the start
            
            # Company
            self.production = data_group["p_hist"][:]
            self.debt = data_group["d_hist"][:]
            self.beta = data_group["beta_hist"][:]
            # Bank
            self.interest_rate = data_group["interest_rate_hist"][:]
            self.interest_rate_free = data_group["interest_rate_hist_free"][:]
            self.d_bank = data_group["d_bank_hist"][:]
            # Attributes
            self.time_steps = data_group.attrs["time_steps"]
            self.N = data_group.attrs["N"]
            
        self.time_values = np.arange(self.time_steps)
        self.money = np.zeros((self.time_steps, self.N))


    def plot_companies(self, N_plot):
        """Plot the first N_plot companies.
        """
        # Load data and create time values array
        production_plot = self.production[: N_plot, :].T
        debt_plot = self.debt[: N_plot, :].T

        
        # Plot averages single axis
        fig, (ax0, ax1) = plt.subplots(nrows=2)
        # mask = slice(0, self.time_steps)
        # ax0.plot(self.time_values[mask], production_plot[mask],)
        # ax1.plot(self.time_values[mask], debt_plot[mask])
        ax0.plot(self.time_values, production_plot)
        ax1.plot(self.time_values, debt_plot)

        # Figure setup
        ax0.set(ylabel="$", title="Production", yscale="log")
        ax1.set(ylabel="$", title="Debt")
        
        # Display parameter values
        self._add_parameters_text(ax0)
        
        fig.suptitle(f"First {N_plot} companies", fontsize=15, fontstyle="italic")
        # Save figure
        self._save_fig(fig, "singlecompanies")
        if self.show_plots: plt.show()


    def plot_means(self):
        # Calculate means of debt and production
        production_mean = np.mean(self.production, axis=0)
        debt_mean = np.mean(self.debt, axis=0)
        
        # Create figure
        fig, (ax, ax1, ax2) = plt.subplots(nrows=3)
        
        # ax 0: Production
        ax.plot(self.time_values, production_mean, label=r"$p$")
        ax.set(ylabel="log $", title="Company mean of production", yscale="log")
        
        # ax 1: Debt
        # Mask for positive and negative values of debt
        positive_mask = debt_mean >= 0
        negative_mask = debt_mean < 0
        
        # Plot positive and negative values separately
        ax1.plot(self.time_values[positive_mask], debt_mean[positive_mask], ".", label=r"$d$ (positive)", color="green")
        ax1.plot(self.time_values[negative_mask], np.abs(debt_mean[negative_mask]), ".", label=r"$d$ (negative, abs)", color="red")
        ax1.set(ylabel="$", title="Company mean of debt", yscale="log")
        
        # ax 2: Bank
        # Mask for positive and negative values of bank debt
        positive_mask = self.d_bank >= 0
        negative_mask = self.d_bank < 0
        ax2.plot(self.time_values[positive_mask], self.d_bank[positive_mask], ".", color="green", label=r"$d_{\text{B}}$ (positive)")  # Positive values
        ax2.plot(self.time_values[negative_mask], np.abs(self.d_bank[negative_mask]), ".", color="red", label=r"$d_{\text{B}}$ (negative, abs)")  # abs negative
        ax2.set(ylabel="Log $", title="Bank debt", yscale="log")
            
        # Grid
        ax.grid()
        ax1.grid()
        ax2.grid()
        
        # Legend
        ax1.legend(fontsize=7)
        ax2.legend(fontsize=7)
        
        # Display parameters
        self._add_parameters_text(ax)
        
        # Save and show figure
        self._save_fig(fig, "means")
        if self.show_plots: plt.show()
    
    
    def plot_beta_evolution(self):
        # Preprocess
        beta_min, beta_max = self.beta.min(), self.beta.max()
        beta_mean = np.mean(self.beta, axis=0)
        
        # Create fig
        fig, (ax, ax1, ax2) = plt.subplots(nrows=3)
        
        # Plots
        # ax: Interest rate and adjusted interest rate
        ax.plot(self.time_values, self.interest_rate_free, "-", label=r"$r_f$")
        ax.plot(self.time_values, self.interest_rate, "--", label=r"$r$")
        # ax 1: Mean beta
        ax1.plot(self.time_values, beta_mean, color="black")
        # ax 2: Beta evolution
        im = ax2.imshow(self.beta, cmap="hot", vmin=beta_min, vmax=beta_max, aspect="auto", origin="lower")
        
        # Colorbar
        fig.colorbar(im)
        
        # Axis setup
        ax.set(ylabel="Interest rate", xlabel="Time", title="Free and adjusted interest rate", xlim=(0, self.time_steps),)# yscale="log")
        ax1.set(ylabel=r"Mean $\beta$", xlabel="Time", title=r"Company Mean $\beta$", xlim=(0, self.time_steps))
        ax2.set(ylabel="Companies", xlabel="Time", title=r"$\beta$ evolution")
        # Ticks
        ax.set_yticks([1e-2, 1e-1, 1e0])
        # Grid
        ax.grid()
        ax1.grid()
        # Legend
        ax.legend(ncols=2, bbox_to_anchor=(0.5, 0.75), loc="lower center", fontsize=6)
        # Parameters
        self._add_parameters_text(ax)
        # Save and show fig
        self._save_fig(fig, name="beta_evolution")
        if self.show_plots: plt.show()
        
    
if __name__ == "__main__":
    group_name = first_group_params
    show_plots = True
    
    bank_vis = BankVisualization(group_name, show_plots)
    
    bank_vis.plot_companies(N_plot=4)
    bank_vis.plot_means()
    bank_vis.plot_beta_evolution()
    