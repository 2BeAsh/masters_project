import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import h5py

# My files
import general_functions
from p_limit_master import dir_path_output, dir_path_image, group_name

class BankVisualization():
    def __init__(self, group_name, show_plots, add_parameter_text_to_plot):
        self.group_name = group_name
        self.show_plots = show_plots
        self.add_parameter_text_to_plot = add_parameter_text_to_plot
        
        self.dir_path_image = dir_path_image
        self.data_path = dir_path_output / "p_limit_simulation_data.h5"
        
        # Load data
        with h5py.File(self.data_path, "r") as file:
            data_group = file[group_name]
            self.filename = data_group.name.split("/")[-1]
            
            # Company
            self.p = data_group["p"][:]
            self.d = data_group["d"][:]
            
            # Bank
            self.interest_rate_free = data_group["interest_rate_free"][:]
            self.interest_rate = data_group["interest_rate"][:]
            
            # Other
            self.went_bankrupt_list = data_group["went_bankrupt"][:]
        
            # Attributes
            self.alpha = data_group.attrs["alpha"]
            
        self.N = self.p.shape[0]
        time_steps = self.p.shape[1]
        self.time_values = np.arange(time_steps)


    def _save_fig(self, figure, name):
        figname = Path.joinpath(self.dir_path_image, name + "_" + self.filename + ".png")
        figure.savefig(figname)
        
    
    def _save_anim(self, animation, name):
        anim_name = Path.joinpath(self.dir_path_image, name + "_" + self.filename + ".mp4")
        animation.save(anim_name, fps=30, writer="ffmpeg")

    
    def plot_companies(self, N_plot):
        p_plot = self.p[: N_plot, :].T
        d_plot = self.d[: N_plot, :].T
        
        fig, (ax, ax1) = plt.subplots(nrows=2)
        ax.plot(self.time_values, p_plot)
        ax1.plot(self.time_values, d_plot)
        
        ax.set(ylabel="Log Price", title="Production", yscale="log")
        ax1.set(ylabel="Price", title="Debt")
        
        # Display parameter values
        if self.add_parameter_text_to_plot: general_functions.add_parameters_text(group_name, ax)
        
        # Save figure
        self._save_fig(fig, "singlecompanies")
        if self.show_plots: plt.show()
        
        
    def plot_means(self):
        p_mean = np.mean(self.p, axis=0)
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
        ax.plot(self.time_values, p_mean, label=r"$p$")
        
        # Setup
        ax.set(xlabel="Time", ylabel="Log Price", title="Mean values", yscale="log")
        ax.grid()
        ax.legend(ncols=3, bbox_to_anchor=(0.5, 1), loc="lower center", fontsize=10)
        
        # Display parameters
        if self.add_parameter_text_to_plot: general_functions.add_parameters_text(group_name, ax)
        # Save and show
        self._save_fig(fig, "means")
        if self.show_plots: plt.show()
        
        
    def plot_number_of_bankruptcies(self):
        
        # Create figure
        fig, ax = plt.subplots()
        ax.plot(self.time_values, self.went_bankrupt_list / self.N, label="Bankruptcies")
        
        # Setup
        ax.set(xlabel="Time", ylabel="Fraction", title="Fraction of companies went bankruptcies")
        
        # Display parameters
        if self.add_parameter_text_to_plot: general_functions.add_parameters_text(group_name, ax)
        # Save and show
        self._save_fig(fig, "bankruptcies")
        if self.show_plots: plt.show()
        
        
    def plot_interest_rates(self):
        # Plot the free interest rate and the interest rate together
        fig, ax = plt.subplots()
        ax.plot(self.time_values, self.interest_rate_free, ls="--", label="Interest rate free")
        ax.plot(self.time_values, self.interest_rate, label="Interest rate")
        ax.axhline(y=self.alpha, color="grey", linestyle="--", label=r"$\alpha$")
        ax.set(xlabel="Time", ylabel="Interest rate")
        ax.grid()
        ax.legend(bbox_to_anchor=(0.5, 1), loc="lower center", fontsize=10, ncols=3)
        
        # Add parameters text
        if self.add_parameter_text_to_plot: general_functions.add_parameters_text(group_name, ax)
        # Save and show
        self._save_fig(fig, "interest_rates")
        if self.show_plots: plt.show()

        
if __name__ == "__main__":
    show_plots = True
    add_parameter_text_to_plot = True
    bank_vis = BankVisualization(group_name, show_plots, add_parameter_text_to_plot)
    
    print("Started plotting")
    bank_vis.plot_companies(4)
    bank_vis.plot_means()
    bank_vis.plot_number_of_bankruptcies()
    bank_vis.plot_interest_rates()
    
    print("Finished plotting")