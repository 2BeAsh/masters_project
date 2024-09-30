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
from bank_master_no_money import first_group_params, dir_path_output, dir_path_image


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


    def plot_means(self):
        # Calculate means of debt and production
        production_mean = np.mean(self.production, axis=0)
        debt_mean = np.mean(self.debt, axis=0)
        
        # Create figure
        fig, ax = plt.subplots()
        
        # Plot means
        ax.plot(self.time_values, production_mean, label=r"$p$")
        ax.plot(self.time_values, debt_mean, label=r"$d$")
        ax.plot(self.time_values, self.d_bank, label=r"$d_{\text{B}}$")
        
        # Axis setup
        ax.set(xlabel="Time", ylabel="Mean Value", title="Company means of production and debt")
        
        # Legend
        ax.legend(ncols=3, bbox_to_anchor=(0.5, 0.9), loc='lower center')
        
        # Grid
        ax.grid()
        
        # Display parameters
        self._add_parameters_text(ax)
        
        # Save and show figure
        self._save_fig(fig, "means")
        if self.show_plots: plt.show()
        
        
if __name__ == "__main__":
    group_name = first_group_params
    show_plots = True
    
    bank_vis = BankVisualization(group_name, show_plots)
    
    # bank_vis.simple_plot()
    bank_vis.plot_means()