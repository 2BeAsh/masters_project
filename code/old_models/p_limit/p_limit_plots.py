import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from pathlib import Path
import h5py

# My files
import general_functions
from p_limit_master import dir_path_output, dir_path_image, group_name

class BankVisualization(general_functions.PlotMethods):
    def __init__(self, group_name, show_plots, add_parameter_text_to_plot):
        super().__init__(group_name)
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
            self.loan_size = data_group["loan_size"][:]
            
            # Other
            self.went_bankrupt_list = data_group["went_bankrupt"][:]
        
            # Attributes
            self.alpha = data_group.attrs["alpha"]
            
        self.N = self.p.shape[0]
        self.time_steps = self.p.shape[1]
        self.time_values = np.arange(self.time_steps)

    
    def plot_companies(self, N_plot):
        p_plot = self.p[: N_plot, :].T
        d_plot = self.d[: N_plot, :].T
        
        fig, (ax, ax1) = plt.subplots(nrows=2)
        ax.plot(self.time_values, p_plot)
        ax1.plot(self.time_values, d_plot)
        
        ax.set(ylabel="Log Price", title="Production", yscale="log")
        ax1.set(ylabel="Price", title="Debt")
        
        # Display parameter values
        if self.add_parameter_text_to_plot: self._add_parameters_text(ax)
        
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
        self._add_legend(ax, ncols=3, y=0.9)
        
        # Display parameters
        if self.add_parameter_text_to_plot: self._add_parameters_text(ax)
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
        if self.add_parameter_text_to_plot: self._add_parameters_text(ax)
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
        if self.add_parameter_text_to_plot: self._add_parameters_text(ax)
        # Save and show
        self._save_fig(fig, "interest_rates")
        if self.show_plots: plt.show()
        
        
    def size_distribution(self):
        # Log histogram of p values, and a histogram of d values where positive and abs(negative) are plotted seperately
        # Final time value of production and debt
        
        production_final = self.p[:, -1]  # Production minimum value is 1
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
        
        
    def loan_size_over_time(self):
        
        # Create fig
        fig, (ax, ax1, ax2) = plt.subplots(nrows=3)
        
        xlim = (0, self.time_steps+1)
        
        # Ax 0: Loan size imshow
        im = ax.imshow(self.loan_size, aspect="auto", origin="lower", extent=[0, self.time_steps, 0, self.N], norm=SymLogNorm(linthresh=1), cmap="magma")
        ax.set(ylabel="N", xlabel="Time", title="Loan size", xlim=xlim)

        # Ax 1: Mean loan size
        ax1.plot(self.time_values, np.mean(self.loan_size, axis=0))
        ax1.set(xlabel="Time", ylabel="Log Mean loan size", title="Mean loan size over time", yscale="symlog", xlim=xlim)
        plt.colorbar(im)
        
        # Ax 2: Loan size of four companies
        # ax2.plot(self.time_values, self.loan_size[0, :], label="Company 0")
        # ax2.plot(self.time_values, self.loan_size[10, :], label="Company 10")
        # ax2.plot(self.time_values, self.loan_size[20, :], label="Company 20")
        # ax2.plot(self.time_values, self.loan_size[30, :], label="Company 30")
        ax2.plot(self.time_values, self.loan_size[:30, :].T, label="Company 30")
        ax2.set(xlabel="Time", ylabel="Log Loan size", title="Loan size of four companies", xlim=xlim, yscale="log")
        
        # Parameters
        if self.add_parameter_text_to_plot: self._add_parameters_text(ax)
        # Save
        self._save_fig(fig, "loan_size")
        if self.show_plots: plt.show()

        
if __name__ == "__main__":
    show_plots = True
    add_parameter_text_to_plot = True
    bank_vis = BankVisualization(group_name, show_plots, add_parameter_text_to_plot)
    
    print("Started plotting")
    # bank_vis.plot_companies(4)
    bank_vis.plot_means()
    bank_vis.plot_number_of_bankruptcies()
    bank_vis.plot_interest_rates()
    bank_vis.size_distribution()
    bank_vis.loan_size_over_time()
    
    print("Finished plotting")