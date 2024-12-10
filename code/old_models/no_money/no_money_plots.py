import numpy as np
from time import time
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.animation as animation
from pathlib import Path
from tqdm import tqdm
import h5py
import functools

# My files
import general_functions  # Own library, here only used for matplotlib styling
from debt_deflation_plots import DebtDeflationVisualization
from no_money_master import first_group_params, dir_path_output, dir_path_image


class BankVisualization(DebtDeflationVisualization):
    def __init__(self, group_name, show_plots, add_parameter_text_to_plot):
        super().__init__(group_name, show_plots)
        
        
        # Variable instances
        self.group_name = group_name
        self.show_plots = show_plots
        self.add_parameter_text_to_plot = add_parameter_text_to_plot
        
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
            # Other
            self.went_bankrupt = data_group["went_bankrupt"][:]
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
        if self.add_parameter_text_to_plot: self._add_parameters_text(ax0)
        
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
        if self.add_parameter_text_to_plot: self._add_parameters_text(ax)
        
        # Save and show figure
        self._save_fig(fig, "means")
        if self.show_plots: plt.show()


    def plot_went_bankrupt(self):
        fig, ax = plt.subplots()
        ax.plot(self.time_values, self.went_bankrupt/self.N)
        ax.set(ylabel="Fraction of companies", xlabel="Time", title="Fraction of companies that went bankrupt")
        ax.grid()
        if self.add_parameter_text_to_plot:  self._add_parameters_text(ax)
        self._save_fig(fig, "went_bankrupt")
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
        ax.set(ylabel="Log Interest rate", xlabel="Time", title="Free and adjusted interest rate", xlim=(0, self.time_steps), yscale="log", yticks=[1e-3, 1e-2, 1e-1])
        ax1.set(ylabel=r"Log Mean $\beta$", xlabel="Time", title=r"Company Mean $\beta$", xlim=(0, self.time_steps), yscale="log")
        ax2.set(ylabel="Companies", xlabel="Time", title=r"$\beta$ evolution")
        # Grid
        ax.grid()
        ax1.grid()
        # Legend
        ax.legend(ncols=2, bbox_to_anchor=(0.5, 0.75), loc="lower center", fontsize=6)
        # Parameters
        if self.add_parameter_text_to_plot:  self._add_parameters_text(ax)
        # Save and show fig
        self._save_fig(fig, name="beta_evolution")
        if self.show_plots: plt.show()
        

    def plot_buying_power(self):
        # Calculate components of buying power
        buyer_part = self.beta * self.production / self.interest_rate - self.debt
        buyer_part_mean = np.mean(buyer_part, axis=0)  # Company mean
        production_mean = np.mean(self.production, axis=0)
        buying_power_norm = self.buying_power
        buying_power_sum_of_parts = buyer_part_mean - production_mean
        
        # Create figure
        fig, ax = plt.subplots()
        
        # Plot buyer part, production and buying power
        ax.plot(self.time_values, buying_power_norm, ls="--", label="Buying power", alpha=0.7)
        ax.plot(self.time_values, buyer_part_mean, ls="-", label=r"Mean $\beta p / r - d$", alpha=0.8)
        ax.plot(self.time_values, production_mean, ls="dotted", label=r"Mean $p$", alpha=0.9)
        ax.plot(self.time_values, buying_power_sum_of_parts, ls="dashdot", label=r"B sum of parts", alpha=0.9)
        
        # Axis setup 
        ax.set(xlabel="Time", ylabel="$", title="Buying power and its components")#, yscale="log")
        ax.grid()
        
        # Legend
        ax.legend(bbox_to_anchor=(0.5, 0.9), loc="lower center", fontsize=10, ncols=3)
        
        # Parameters
        if self.add_parameter_text_to_plot:  self._add_parameters_text(ax)
        
        # save and show figure
        self._save_fig(fig, "buying_power")
        if self.show_plots: plt.show()
        
        
    def plot_buying_power_full(self):
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
        if self.add_parameter_text_to_plot:  self._add_parameters_text(ax)
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
        if self.add_parameter_text_to_plot:  self._add_parameters_text(ax)
        # Show plot and Save figure
        self._save_fig(fig, "buying_power")
        if self.show_plots: plt.show()
    
    
    def histograms(self):
        """Standar histograms of the production and debt at the final time values"""
        # Final time value of production and debt
        production_final = self.production[:, -1]  # Production minimum value is 1
        debt_final = self.debt[:, -1] + 1e-6  # Prevent zero values
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
        ax1.legend(ncols=2, bbox_to_anchor=(0.5, 0.9), loc="lower center")
        ax.grid()
        ax1.grid()
        
        # Parameters text
        if self.add_parameter_text_to_plot:  self._add_parameters_text(ax)
        
        # Save and show figure
        if self.show_plots: plt.show()
        self._save_fig(fig, "hist")


    def animate_histograms(self):
        time_i = time()
        
        debt_positive = self.debt[self.debt > 0]
        debt_negative_abs = np.abs(self.debt[self.debt < 0])

        # Binning
        Nbins = int(np.sqrt(self.time_steps)) 
        p_min, p_max = 1e0, self.production.max()*10
        d_min, d_max = 1e-6, np.abs(self.debt).max()*10
        bins_p = 10 ** np.linspace(np.log10(1e0), np.log10(p_max), Nbins)  # Log x cuts off large values if max range value is not increased
        bins_d = 10 ** np.linspace(np.log10(1e-6), np.log10(d_max), Nbins)
        
        bin_edges_p = 10 ** np.linspace(np.log10(p_min), np.log10(p_max), Nbins)
        bin_edges_d = 10 ** np.linspace(np.log10(d_min), np.log10(d_max), Nbins)
        
        # Create figure
        fig, (ax, ax1) = plt.subplots(ncols=2)
            
        # Figure setup        
        fig, (ax, ax1) = plt.subplots(ncols=2)
        
        # Production
        _, _, bar_container_p = ax.hist(self.production[:, 0], bin_edges_p)  # Initial histogram 
        ax.set(xlim=(bin_edges_p[0], bin_edges_p[-1]), 
               title="Time = 0", xscale="log")

        # Debt
        _, _, bar_container_d = ax1.hist(self.debt[:, 0], bin_edges_d)  # Initial histogram 
        ax.set(xlim=(bin_edges_d[0], bin_edges_d[-1]), 
               title="Time = 0", xscale="log")
        
        # Text
        if self.add_parameter_text_to_plot:  self._add_parameters_text(ax)

        def animate(i, bar_container_p, bar_container_d):
            """Frame animation function for creating a histogram."""
            # Production
            data_p = self.production[:, i]
            n_p, _ = np.histogram(data_p, bin_edges_p)
            for count, rect in zip(n_p, bar_container_p.patches):
                rect.set_height(count)
            
            # Debt
            data_d = debt_positive[:, i]
            n_d, _ = np.histogram(data_d, bin_edges_d)
            for count_d, rect_d in zip(n_d, bar_container_d.patches):
                rect_d.set_height(count_d)
            
            # Title
            ax.set_title(f"Time = {i}")
            return bar_container_p.patches + bar_container_d.patches
        
        # Create the animation
        # anim = functools.partial(animate, bar_container=bar_container)
        # ani = animation.FuncAnimation(fig, anim, frames=self.time_steps, interval=1)
        
        # Save animation
        # time_create_ani = time()  # Record time
        # animation_name = Path.joinpath(self.dir_path_image, "size_distribution_animation_" + self.filename + ".mp4")
        # ani.save(animation_name, fps=30)
        
        # Display times
        # time_save_ani = time()
        # print("Time creating animation: \t", time_create_ani - time_i)
        # print("Time saving animation: \t", time_save_ani - time_create_ani)
            
    
if __name__ == "__main__":
    group_name = first_group_params
    show_plots = True
    add_parameter_text_to_plot = True
    bank_vis = BankVisualization(group_name, show_plots, add_parameter_text_to_plot)
    
    print("Started plotting")
    bank_vis.plot_companies(N_plot=4)
    bank_vis.plot_means()
    bank_vis.plot_went_bankrupt()
    bank_vis.plot_beta_evolution()
    # bank_vis.plot_buying_power()
    bank_vis.histograms()
    
    print("Finished plotting")