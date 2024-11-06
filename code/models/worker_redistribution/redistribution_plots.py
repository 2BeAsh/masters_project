import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from pathlib import Path
import h5py
import functools
import matplotlib.animation as animation


# My files
import general_functions
from redistribution_master import dir_path_output, dir_path_image, group_name

class BankVisualization(general_functions.PlotMethods):
    def __init__(self, group_name, show_plots, add_parameter_text_to_plot):
        super().__init__(group_name)
        self.group_name = group_name
        self.show_plots = show_plots
        self.add_parameter_text_to_plot = add_parameter_text_to_plot
        
        # Check if the path to the image folder exists, otherwise create it
        dir_path_image.mkdir(parents=True, exist_ok=True)
        self.dir_path_image = dir_path_image
        self.data_path = dir_path_output / "redistribution_simulation_data.h5"
        
        # Load data
        with h5py.File(self.data_path, "r") as file:
            data_group = file[group_name]
            self.filename = data_group.name.split("/")[-1]
            
            # Company
            self.w = data_group["w"][:]
            self.m = data_group["m"][:] 
            self.d = data_group["d"][:]
            self.s = data_group["s"][:]
            
            # System
            self.interest_rate = data_group["interest_rate"][:]
            self.went_bankrupt = data_group["went_bankrupt"][:]
            self.unemployed = data_group["unemployed"][:]
            self.system_money_spent = data_group["system_money_spent"][:]
        
            # Attributes
            self.sigma = data_group.attrs["salary_increase"]
            self.W = data_group.attrs["W"]
            
        self.N = self.w.shape[0]
        self.time_steps = self.w.shape[1]
        self.time_values = np.arange(self.time_steps)

    
    def plot_companies(self, N_plot):
        w_plot = self.w[: N_plot, :].T
        m_plot = self.m[: N_plot, :].T
        d_plot = self.d[: N_plot, :].T
        
        fig, (ax, ax1, ax2) = plt.subplots(nrows=3)

        ax.plot(self.time_values, w_plot)
        ax.set(ylabel="Number of workers", title="Workforce")
        ax.grid()

        ax1.plot(self.time_values, m_plot)
        ax1.set(ylabel="Number of machines", title="Machines", xlabel="Time")
        ax1.grid()

        ax2.plot(self.time_values, d_plot)
        ax2.set(ylabel="Price", title="Debt", xlabel="Time")
        ax2.grid()
        
        # Display parameter values
        if self.add_parameter_text_to_plot: self._add_parameters_text(ax)
        
        # Save figure
        self._save_fig(fig, "singlecompanies")
        if self.show_plots: plt.show()
        
        
    def plot_means(self):
        w_mean = np.mean(self.w, axis=0)
        d_mean = np.mean(self.d, axis=0)
        m_mean = np.mean(self.m, axis=0)
        
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

        # Machinery
        ax.plot(self.time_values, m_mean, label=r"$m$")
        
        # Setup
        ax.set(xlabel="Time", ylabel="Log Price", title="Mean values", yscale="log")
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
        ax.set(ylabel="Interest rate")
        ax.grid()
        ax.legend(bbox_to_anchor=(0.5, 1), loc="lower center", fontsize=10, ncols=3)
        
        # ax1 bankruptcies
        ax1.plot(self.time_values, self.went_bankrupt / self.N, label="Bankruptcies")
        ax1.set(xlabel="Time", ylabel="Bankrupt fraction", title="Bankruptcies")
        ax1.grid()
        
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
        
        
    def plot_salary(self):
        fig, (ax1, ax2) = plt.subplots(nrows=2)
        
        # All companies' salary over time
        # im = ax.imshow(self.s, cmap="magma")
        # ax.set(xlabel="Time", ylabel="Company", title="Salary over time")
        # fig.colorbar(im)
        
        # ax1 - Mean salary over time
        ax1.plot(self.time_values, np.mean(self.s, axis=0))
        ax1.set(xlabel="Time", ylabel="Price", title="Mean salary over time", xlim=(0, self.time_steps))
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
        max_val = w_final.max()
        bins = np.arange(0, max_val +1 , 1)
        
        fig, ax  = plt.subplots()
        counts, edges, _ = ax.hist(w_final, bins=bins, label=r"$w$", alpha=0.7)
        ax.set(xlabel="Company workforce", ylabel="Counts", title="Production capacity distributions at final time")
        ax.grid()
        
        # xticks
        number_of_xticks = 10
        xticks = np.linspace(0, max_val, number_of_xticks, dtype=int)
        ax.set_xticks(xticks)
        
        # Legend
        self._add_legend(ax, ncols=1, y=0.9)
        
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
        
        fig, (ax0, ax1) = plt.subplots(nrows=2)
        
        # ax0 - System money spent
        ax0.plot(self.time_values, self.system_money_spent)
        ax0.set(ylabel="Log $", title="System money spent", yscale="log")
        ax0.grid()
        
        # ax1 - Fraction employed
        ax1.plot(self.time_values, 1 - self.unemployed / self.W)
        ax1.set(ylabel="Fraction", title="Fraction employed")
        ax1.grid()
        
        # # ax2 - Inflation
        # inflation = np.diff(self.system_money_spent) / np.maximum(self.system_money_spent[:-1], 1e-5) * 100  # Percent. Max to prevent div by 0
        # ax2.plot(self.time_values[2:], inflation[1:])
        # ax2.set(ylabel="Percent", title="Inflation", xlabel="Time", yscale="symlog")
        # ax2.grid()
        
        # Add parameters text
        if self.add_parameter_text_to_plot: self._add_parameters_text(ax0)
        
        # Save and show
        self._save_fig(fig, "system_money")
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
        animation_name = Path.joinpath(self.dir_path_image, "workforce_animation" + self.filename + ".mp4")
        ani.save(animation_name, fps=30)
  
  
if __name__ == "__main__":
    show_plots = True
    add_parameter_text_to_plot = True
    bank_vis = BankVisualization(group_name, show_plots, add_parameter_text_to_plot)
    
    print("Started plotting")
    bank_vis.plot_companies(4)
    bank_vis.plot_means()
    bank_vis.plot_interest_rates()
    bank_vis.plot_production_capacity()
    bank_vis.plot_salary()
    bank_vis.plot_system_money()
    
    # bank_vis.animate_size_distribution()

    
    print("Finished plotting")