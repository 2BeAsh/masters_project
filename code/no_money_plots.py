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
            self.buying_power = data_group["buying_power_hist"][:]
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
        fig, ax = plt.subplots()
        
        # ax 0: Production
        ax.plot(self.time_values, production_mean, label=r"Mean $p$")
        ax.set(ylabel="log $", yscale="log", label=r"Mean $d$ (abs negative)")
        
        # ax 1: Debt
        # Mask for positive and negative values of debt
        positive_mask = debt_mean >= 0
        negative_mask = debt_mean < 0
        
        # Plot positive and negative values separately
        ax.plot(self.time_values[positive_mask], debt_mean[positive_mask], ".", label=r"$d$ (positive)", color="green", markersize=1.5)
        ax.plot(self.time_values[negative_mask], np.abs(debt_mean[negative_mask]), ".", label=r"$d$ (negative, abs)", color="red", markersize=1.5)
        
        # Grid
        ax.grid()
        
        # Legend
        ax.legend(ncols=3, bbox_to_anchor=(0.5, 1), loc="lower center", fontsize=10)
        
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
        im = ax2.imshow(self.beta, cmap="hot", vmin=beta_min, vmax=beta_max, aspect="auto", origin="lower", interpolation="none")
        
        # Colorbar
        fig.colorbar(im)
        
        # Axis setup
        ax.set(ylabel="Log Interest rate", xlabel="Time", title="Free and adjusted interest rate", xlim=(0, self.time_steps), yscale="log")
        ax1.set(ylabel=r"Log Mean $\beta$", xlabel="Time", title=r"Company Mean $\beta$", xlim=(0, self.time_steps), yscale="log")
        ax2.set(ylabel="Companies", xlabel="Time", title=r"$\beta$ evolution")
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
        
        
    def plot_buying_power(self):
        """Plot the buying power together with its components i.e. beta, production, interest and debt.
        One subplot has only buying power, one has production and debt, and the last has beta and interest rate.
        """
        # Calculate components of buying power
        positive_part = self.beta * self.production / self.interest_rate
        positive_part_mean = np.mean(positive_part, axis=0)  # Company mean
        debt_mean = np.mean(self.debt, axis=0)

        
        # Create figure
        fig, (ax, ax1) = plt.subplots(nrows=2)
    
        # Plots with negative values are plotted with color-coded line segments
        for i in range(len(self.time_values) - 1):
            # ax0 : buying power and its components
            color = "green" if self.buying_power[i] >= 0 else "red"
            ax.plot(self.time_values[i:i+2], np.abs(self.buying_power[i:i+2]), color=color, linewidth=0.5)
            
            color = "purple" if debt_mean[i] >= 0 else "brown"
            ax.plot(self.time_values[i:i+2], debt_mean[i:i+2], color=color, linewidth=0.5)


        ax.plot(self.time_values, positive_part_mean, label=r"$\beta p / r$", color="black")

        ax.set(ylabel="$", title="Buying power, Mean Production and Debt", yscale="log")
        
        # ax1 1: Beta and interest rate
        ax1.plot(self.time_values, np.mean(self.beta, axis=0), label=r"$\beta$")
        ax1.plot(self.time_values, self.interest_rate, label=r"$r$")
        ax1.set(ylabel="Log $", title="Beta and interest rate", yscale="log")
        # ax1.set_yticks([1e-3, 1e-2, 1e-1, 1e0])
        
        # Grids for all axes
        ax.grid()
        ax1.grid()
        
        # Legends for all axes
        ax.legend()
        ax1.legend()
        
        # Parameter values
        self._add_parameters_text(ax)
        # Show plot and Save figure
        self._save_fig(fig, "buying_power")
        if self.show_plots: plt.show()
        return
    
        fig, (ax, ax1, ax2) = plt.subplots(nrows=3)
        
        # ax 0: Buying power
        # Mask for positive and negative values of buying power
        positive_mask_buying_power = self.buying_power >= 0
        negative_mask_buying_power = self.buying_power < 0
        
        # Plot positive and negative values separately for buying power
        ax.plot(self.time_values[positive_mask_buying_power], self.buying_power[positive_mask_buying_power], ".", markersize=1, label="Buying power (positive)", color="green")
        ax.plot(self.time_values[negative_mask_buying_power], np.abs(self.buying_power[negative_mask_buying_power]), ".", markersize=1, label="Buying power (negative, abs)", color="red")
        
        ax.set(ylabel="$", title="Buying power", yscale="log")
        
        # ax1 1: Production and debt
        # Mask for positive and negative values of positive_part_mean
        positive_mask_positive_part = positive_part_mean >= 0
        negative_mask_positive_part = positive_part_mean < 0
        
        # Mask for positive and negative values of debt_mean
        positive_mask_debt = debt_mean >= 0
        negative_mask_debt = debt_mean < 0
        
        # Plot positive and negative values separately for positive_part_mean
        ax1.plot(self.time_values[positive_mask_positive_part], positive_part_mean[positive_mask_positive_part], ".", markersize=1.5, label=r"$\beta p / r$ (positive)", color="darkgreen")
        ax1.plot(self.time_values[negative_mask_positive_part], np.abs(positive_part_mean[negative_mask_positive_part]), ".", markersize=1.5, label=r"$\beta p / r$ (negative, abs)", color="darkred")
        
        # Plot positive and negative values separately for debt_mean
        ax1.plot(self.time_values[positive_mask_debt], debt_mean[positive_mask_debt], "*", markersize=1.5, label="Debt (positive)", color="green")
        ax1.plot(self.time_values[negative_mask_debt], np.abs(debt_mean[negative_mask_debt]), "*", markersize=1.5, label="Debt (negative, abs)", color="red")
        
        ax1.set(ylabel="$", title="Mean Production and debt", yscale="log")
        
        # ax2 2: Beta and interest rate
        ax2.plot(self.time_values, self.beta.mean(axis=0), label=r"$\beta$")
        ax2.plot(self.time_values, self.interest_rate, label=r"$r$")
        ax2.set(ylabel="Log $", title="Beta and interest rate", yscale="log")
        ax2.set_yticks([1e-3, 1e-2, 1e-1, 1e0])
        
        # Grids for all axes
        ax.grid()
        ax1.grid()
        ax2.grid()
        
        # Legends for all axes
        ax.legend()
        ax1.legend()
        ax2.legend()
        
        # Parameter values
        self._add_parameters_text(ax)
        # Show plot and Save figure
        self._save_fig(fig, "buying_power")
        if self.show_plots: plt.show()
    
    
    def histograms(self):
        """Histograms of production, debt
        """
        
        # Bin data of final times steps. 
        p_final = self.production[:, -1]
        d_final = self.debt[:, -1]
        
        Nbins = int(np.sqrt(self.time_steps))
        p_min = self.production.min()
        p_max = self.production.max()
        bins_log = 10 ** np.linspace(np.log10(p_min), np.log10(p_max), Nbins)
        
        # Create figure 
        fig, (ax1, ax2) = plt.subplots(nrows=2)

        # Plot histogram for p_final
        ax1.hist(p_final, bins=bins_log)
        ax1.set(xlabel="Production", ylabel="Frequency", xscale="log")

        # Plot histogram for d_final
        ax2.hist(d_final, bins=Nbins)
        ax2.set(xlabel='Debt', ylabel='Frequency')

        # Adjust layout and show/save figure
        self._save_fig(fig, "histograms")
        if self.show_plots: plt.show()
    
    
if __name__ == "__main__":
    group_name = first_group_params
    show_plots = True
    
    bank_vis = BankVisualization(group_name, show_plots)
    
    # bank_vis.plot_companies(N_plot=4)
    # bank_vis.plot_means()
    # bank_vis.plot_beta_evolution()
    # bank_vis.plot_buying_power()
    bank_vis.histograms()
    
    print("Finished plotting")