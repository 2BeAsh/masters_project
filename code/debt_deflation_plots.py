import numpy as np
from time import time
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.animation as animation
import functools
from pathlib import Path
# My files
import general_functions  # Own library, here only used for matplotlib styling
from debt_deflation_well_mixed import filename_parameter_addon
from debt_deflation_1d import filename_parameter_addon_1d
from debt_deflation_master import real_interest_rate


class DebtDeflationVisualization():
    def __init__(self, filename, show_plots=False):
        """Collection of plotting methods

        Args:
            filename (str): Filename addon 
            show_plots (bool, optional): Show plots. If False, only saves. Defaults to False.
        """
        # Local paths for saving files.
        self.filename = filename
        file_path = Path(__file__)
        self.dir_path = file_path.parent
        self.dir_path_output = Path.joinpath(self.dir_path, "output")
        
        # Get image path from filename
        filename_split = filename.split("_")
        image_sufix = filename_split[-1] + "/"
        self.dir_path_image = Path.joinpath(self.dir_path, "image", image_sufix)

        # If want to display/show plots or not
        self.show_plots = show_plots
        

    def _display_parameters(self):
        # Split at underscores "_"
        filename_list = self.filename.split("_")
        
        # Combine strings and add newlines
        filename_comb = ""
        for arg_name in filename_list:
            filename_comb += arg_name + "\n"
        return filename_comb


    def _add_parameters_text(self, axis, x=0.1, y=0.8):
        display_parameters_str = self._display_parameters()
        axis.text(x=x, y=y, s=display_parameters_str, transform=axis.transAxes, horizontalalignment='left', verticalalignment='center', fontsize=9)
        

    def _load_data(self, parameter_change=False) -> tuple:
        """Loads data from .npy file. 

        Args:
            parameter_change (bool, optional): Load parameter change data. Defaults to False.

        Returns:
            tuple: Data variables
        """
        if parameter_change:
            filename = Path.joinpath(self.dir_path_output, "parameter_change_" + self.filename + ".npy")
            data_all = np.load(filename)
            production = data_all[:, :, 0]
            debt = data_all[:, :, 1]
            money = data_all[:, :, 2] 
            
            r_vals = data_all[:, :, 3]
            r_vals_true = r_vals[:, 0]
            
            self.time_steps = np.shape(production)[1]
            return production, debt, money, r_vals_true
        
        else:
            filename = Path.joinpath(self.dir_path_output, self.filename + ".npy")
            data_all = np.load(filename)
            production = data_all[:, :, 0]
            debt = data_all[:, :, 1]
            money = data_all[:, :, 2] 
            self.time_steps = np.shape(production)[1]
            return production, debt, money
    
    
    def _save_fig(self, figure, name):
        figname = Path.joinpath(self.dir_path_image, name + "_" + self.filename + ".png")
        figure.savefig(figname)
        
    
    def _save_anim(self, animation, name):
        anim_name = Path.joinpath(self.dir_path_image, name + "_" + self.filename + ".mp4")
        animation.save(anim_name, fps=30, writer="ffmpeg")
        
    
    def plot_means(self, scale="linear"):
        # Load data and create time values array
        production, debt, money = self._load_data()
        
        # Averages
        production_mean = np.mean(production, axis=0)
        debt_mean = np.mean(debt, axis=0)
        money_mean = np.mean(money, axis=0)
        time_values = np.arange(0, self.time_steps)
        
        if scale == "log":
            if (production_mean < 1).any(): print("(Plot means) Detected production values below 1!")
            debt_mean = np.maximum(debt_mean, 1e-1)  # Debt can go to zero
            money_mean = np.maximum(money_mean, 1e-1)  # Money can go negative!!!
        
        fig, ax = plt.subplots()
        
        # Plot means
        ax.plot(time_values, production_mean, label="Company production")
        ax.plot(time_values, debt_mean, label="Debt")
        ax.plot(time_values, money_mean, "--", label="Money")
        
        # Axis setup
        ax.set(xlabel="Time", ylabel="$", title="Mean values", yscale=scale)
        
        # Legend
        legend_elements = [Line2D([], [], color="rebeccapurple", label="Production"),
                           Line2D([], [], color="firebrick", label="Debt"),
                           Line2D([], [], color="black", label="Money"),]
        ax.legend(handles=legend_elements, ncols=3, bbox_to_anchor=(0.5, 0.9), loc="lower center")

        # Grid
        ax.grid()

        # Display parameters
        self._add_parameters_text(ax)

        # Save figure
        figname = Path.joinpath(self.dir_path_image, f"means_" + self.filename + ".png")
        plt.savefig(figname)
        if self.show_plots: plt.show()


    def plot_companies(self, N_plot, scale="linear"):
        """Plot the first N_plot companies.
        """
        # Load data and create time values array
        production, debt, money = self._load_data()
        production_plot = production[:N_plot, :].T
        debt_plot = debt[: N_plot, :].T
        money_plot = money[: N_plot, :].T
        time_values = np.arange(0, self.time_steps)

        if scale == "log":
            if (production_plot < 1).any(): print("(Single companies) Detected production values below 1!")
            debt_plot = np.maximum(debt_plot, 1e-1)  # Debt can go to zero
        
        # Plot averages single axis
        fig, (ax0, ax1, ax2) = plt.subplots(nrows=3)
        mask = slice(0, self.time_steps)
        ax0.plot(time_values[mask], production_plot[mask],)
        ax1.plot(time_values[mask], debt_plot[mask])
        ax2.plot(time_values[mask], money_plot[mask] )

        # Figure setup
        ax0.set(ylabel="$", title="Production", yscale=scale)
        ax1.set(ylabel="$", title="Debt", yscale=scale)
        ax2.set(ylabel="$", xlabel="Time", title="Money", yscale="linear")  # Money can go negative
        
        # Display parameter values
        self._add_parameters_text(ax0)
        
        fig.suptitle(f"First {N_plot} companies", fontsize=15, fontstyle="italic")
        # Save figure
        figname = Path.joinpath(self.dir_path_image, f"single_companies_" + self.filename + ".png")
        plt.savefig(figname)
        if self.show_plots: plt.show()


    def final_time_size_dist(self):
        production, debt, money = self._load_data()
        Nbins = int(np.sqrt(self.time_steps))
        p_min = production.min()
        p_max = production.max()
        bins = 10 ** np.linspace(np.log10(p_min), np.log10(p_max), Nbins)
        bin_edges = np.linspace(production.min(), production.max(), Nbins)
        
        production_final = production[:, -1]
        fig, ax = plt.subplots()
        n, _, _ = ax.hist(production_final, bins=bins)
        ax.set(xlabel="Production", title="Final time production distribution", 
               ylabel="Frequency", xscale="log")

        # Parameters text
        self._add_parameters_text(ax)

        # Save figure
        figname = Path.joinpath(self.dir_path_image, f"final_time_dist_" + self.filename + ".png")
        plt.savefig(figname)
        if self.show_plots: plt.show()
        

    def final_time_values(self, scale="log"):
        production, debt, money = self._load_data()
        production_final = production[:, -1]
        debt_final = debt[:, -1]
        money_final = money[:, -1]
        x_vals = np.arange(len(money_final))

        # Since using log, cannot have <= 0 values
        if scale == "log":
            y_min = 1e-3
            money_final = np.maximum(money_final, y_min)
            debt_final = np.maximum(debt_final, y_min)

        # Create figure
        fig, ax = plt.subplots(nrows=3)
        ax_p, ax_d, ax_m = ax
        
        # Lines
        ax_p.plot(x_vals, production_final, ".", color="rebeccapurple")
        ax_d.plot(x_vals, debt_final, "x", color="firebrick")
        ax_m.plot(x_vals, money_final, "*", color="black")
        
        # Axes setup
        # Legend
        legend_elements = [Line2D([], [], color="rebeccapurple", marker=".", ls="none", label="Production"),
                        Line2D([], [], color="firebrick", marker="x", ls="none",label="Debt"),
                        Line2D([], [], color="black", marker="*", ls="none", label="Money"),]
        ax_p.legend(handles=legend_elements, ncols=3, bbox_to_anchor=(0.5, 0.99), loc="lower center")
        # Scale
        ax_p.set_yscale(scale)
        ax_d.set_yscale(scale)
        ax_m.set_yscale(scale)
        # Axis set
        N = np.shape(production)[0]
        ax_p.set(xlim=(0, N), ylim=(np.min(production_final), np.max(production_final)), ylabel="Production value")
        ax_d.set(xlim=(0, N), ylim=(np.min(debt_final), np.max(debt_final)), ylabel="Debt")
        ax_m.set(xlim=(0, N), ylim=(np.min(money_final), np.max(money_final)), xlabel="Company Number", ylabel="Money")
        # Grid
        ax_p.grid()
        ax_d.grid()
        ax_m.grid()
        # Ticks
        ax_p.set_xticks(ticks=x_vals, labels=x_vals, fontsize=3)
        ax_d.set_xticks(ticks=x_vals, labels=x_vals, fontsize=3)
        ax_m.set_xticks(ticks=x_vals, labels=x_vals, fontsize=3)

        # Display paraemters
        self._add_parameters_text(ax_p, x=0.05, y=0.7)
        
        # Save figure
        figname = Path.joinpath(self.dir_path_image, f"final_time_values" + self.filename + ".png")
        plt.savefig(figname)
        if self.show_plots: plt.show()


    def plot_inflation_rate(self) -> None:
        # Load data
        filename = Path.joinpath(self.dir_path_output, "inflation_" + self.filename + ".npy")
        inflation_rates = np.load(filename)
        
        # Create figure
        fig, ax = plt.subplots()
        ax.plot(np.arange(self.time_steps), inflation_rates, label=r"$\pi$")
        ax.axhline(real_interest_rate, ls="dashed", c="grey", label="Real interest rate")
        ax.set(xlabel="Time", ylabel="Inflation rate")
        ax.legend(ncols=2, bbox_to_anchor=(0.5, 0.95), loc="lower center")

        # Display paraemters
        self._add_parameters_text(ax, x=0.05, y=0.7)

        # Save and show
        figname = Path.joinpath(self.dir_path_image, f"inflation_rate" + self.filename + ".png")
        plt.savefig(figname)
        if self.show_plots: plt.show()
        

    def animate_size_distribution(self):
        time_i = time()
        # Load data and create time values array
        production, debt, money = self._load_data()
        
        # Bin data
        Nbins = int(np.sqrt(self.time_steps))
        # bin_edges = np.linspace(production.min(), production.max(), Nbins)

        p_min = production.min()
        p_max = production.max()
        bin_edges = 10 ** np.linspace(np.log10(p_min), np.log10(p_max), Nbins)
            
        # Figure setup        
        fig, ax = plt.subplots()
        _, _, bar_container = ax.hist(production[:, 0], bin_edges)  # Initial histogram 
        ax.set(xlim=(bin_edges[0], bin_edges[-1]), 
               title="Time = 0", xscale="log")

        # Text
        self._add_parameters_text(ax)

        def animate(i, bar_container):
            """Frame animation function for creating a histogram."""
            # Histogram
            data = production[:, i]
            n, _ = np.histogram(data, bin_edges)
            for count, rect in zip(n, bar_container.patches):
                rect.set_height(count)
            
            # Title
            ax.set_title(f"Time = {i}")
            return bar_container.patches
        
        # Create the animation
        anim = functools.partial(animate, bar_container=bar_container)
        ani = animation.FuncAnimation(fig, anim, frames=self.time_steps, interval=1)
        
        # Save animation
        time_create_ani = time()  # Record time
        animation_name = Path.joinpath(self.dir_path_image, "size_distribution_animation_" + self.filename + ".mp4")
        ani.save(animation_name, fps=30)
        
        # Display times
        time_save_ani = time()
        print("Time creating animation: \t", time_create_ani - time_i)
        print("Time saving animation: \t", time_save_ani - time_create_ani)
        
        
    def animate_values(self, scale="log", on_same_row=False):
        # Store time at initial time to later find the time taken for different parts of the animation
        time_i = time()
        
        # Load data
        production, debt, money = self._load_data()
        N = np.shape(production)[0]
        x_vals = np.arange(N)
        
        # If using logscale, set minimum value, as log(0) = -infty
        y_min = np.min([production, debt, money])
        if scale == "log":
            y_min = 1e-3
            money = np.maximum(money, y_min)
            debt = np.maximum(debt, y_min)
        
        # Initial image and plot setup

        # Create figure
        fig, ax = plt.subplots(nrows=3)
        ax_p, ax_d, ax_m = ax
        # Initial line
        line_prod = ax_p.plot(x_vals, production[:, 0], ".", color="rebeccapurple")[0]
        line_debt = ax_d.plot(x_vals, debt[:, 0], "x", color="firebrick")[0]
        line_money = ax_m.plot(x_vals, money[:, 0], "*", color="black")[0]
        
        # Axes setup
        # Legend
        legend_elements = [Line2D([], [], color="rebeccapurple", marker=".", ls="none", label="Production"),
                        Line2D([], [], color="firebrick", marker="x", ls="none",label="Debt"),
                        Line2D([], [], color="black", marker="*", ls="none", label="Money"),]
        ax_p.legend(handles=legend_elements, ncols=3, bbox_to_anchor=(0.5, 0.9), loc="lower center")
        # Scale
        ax_p.set_yscale(scale)
        ax_d.set_yscale(scale)
        ax_m.set_yscale(scale)
        # Axis set
        ylim_p = (1, np.max(production))
        ylim_d = (np.min(debt), np.max(debt))
        ylim_m = (np.min(money), np.max(money))
        ax_p.set(xlim=(0, N), ylim=ylim_p, ylabel="Production value")
        ax_d.set(xlim=(0, N), ylim=ylim_d, ylabel="Debt")
        ax_m.set(xlim=(0, N), ylim=ylim_m, xlabel="Company Number", ylabel="Money")
        # Grid
        ax_p.grid()
        ax_d.grid()
        ax_m.grid()
        
        # Ticks
        def ticks(axis):
            axis.set_xticks(ticks=x_vals, labels=x_vals, fontsize=3)
            current_locs = axis.get_yticks()
            current_labels = axis.get_yticklabels()
            return current_locs, current_labels
        
        locs_p, labels_p = ticks(ax_p)
        locs_d, labels_d = ticks(ax_d)
        locs_m, labels_m = ticks(ax_m)

        # Parameters text
        self._add_parameters_text(ax_p)
        
        # Line update function
        def animate(i):
            # Set lines
            line_prod.set_ydata(production[:, i])
            line_debt.set_ydata(debt[:, i])
            line_money.set_ydata(money[:, i])
            # Figure setup. Current time and stop yticks from jittering
            fig.suptitle(f"Time = {i}")
            ax_p.set_yticks(locs_p, labels_p)
            ax_d.set_yticks(locs_d, labels_d)
            ax_m.set_yticks(locs_m, labels_m)
            return line_prod, line_debt, line_money
    
        
        # Create animation and save it
        anim = animation.FuncAnimation(fig, animate, interval=1, frames=self.time_steps, blit=True)
        
        time_create_anim = time()  # Record time
        animation_name = Path.joinpath(self.dir_path_image, "value_animation_" + f"{scale}_same_row{on_same_row}_" + self.filename + ".mp4")
        anim.save(animation_name, fps=30)
        
        # Display times
        time_save_anim = time()
        print("Time creating animation: \t", (time_create_anim - time_i) / 60, " min")
        print("Time saving animation: \t", (time_save_anim - time_create_anim) / 60, " min")
        
        
    def animate_mean_under_parameter_change(self):
        # Save time for later speed calculations
        time_i = time()

        # Get the data
        production_means, debt_means, money_means, r_vals = self._load_data(parameter_change=True)        
        time_values = np.arange(0, self.time_steps)

        # Get the number of repeats by dividing total array size with unique array size
        N_repeats = r_vals.size // np.unique(r_vals).size
        
        # Create figure and setup axis        
        fig, ax = plt.subplots()
        # Get limits
        # ymin = np.min([production_means, debt_means, money_means])
        # ymax = np.max([production_means, debt_means, money_means])
        
        ax.set(xlabel="Time", ylabel="$", title="Mean values", 
               xlim=(time_values[0], time_values[-1]), yscale="log")
            #    ylim=(ymin, ymax))
        
        # Legend
        legend_elements = [Line2D([], [], color="rebeccapurple", label="Production"),
                           Line2D([], [], color="firebrick", label="Debt"),
                           Line2D([], [], color="black", label="Money"),]
        ax.legend(handles=legend_elements, ncols=3, bbox_to_anchor=(0.5, 0.9), loc="lower center")

        # Create initial lines
        line_p = ax.plot(time_values, production_means[0, :])[0]
        line_d = ax.plot(time_values, debt_means[0, :])[0]
        # line_m = ax.plot(time_values, money_means[0, :], "--")[0]
        
        # Update function
        def animate(i):
            line_p.set_ydata(production_means[i, :])
            line_d.set_ydata(debt_means[i, :])
            # line_m.set_ydata(money_means[i, :])
            
            # Title. r value, time_steps (and repeats?)
            r_val_i = r_vals[i]
            ax.set_title(label=f"r = {r_val_i}, steps = {i}, {i % N_repeats}/{N_repeats}")
            
            # y limits. Only update ylim when new r value
            if i % N_repeats == 0:
                # Find min and max y value of the next N_repeats values
                ymin = 0.8 * np.min([production_means[i: i + N_repeats, :], debt_means[i: i + N_repeats, :],])# money_means[i: i + N_repeats, :]])  # 0.8 factor to reduce impact of outliers
                ymax = 0.8 * np.max([production_means[i: i + N_repeats, :], debt_means[i: i + N_repeats, :],])# money_means[i: i + N_repeats, :]])
                ax.set_ylim(ymin, ymax)
            ax.set_xlim(time_values[0], time_values[-1])

        # Create animation and save it
        anim = animation.FuncAnimation(fig, animate, interval=1, frames=np.shape(production_means)[0])
        
        time_create_anim = time()  # Record time
        animation_name = Path.joinpath(self.dir_path_image, "parameter_change_animation_" + self.filename + ".mp4")
        anim.save(animation_name, fps=2)
        
        # Display times
        time_save_anim = time()
        print("Parameter change animation:")
        print("Time creating animation: \t", (time_create_anim - time_i)/60, " min")
        print("Time saving animation: \t", (time_save_anim - time_create_anim)/60, " min")
            

if __name__ == "__main__":      
    run_well_mixed = True
    run_1d = False
    run_animations = True
    show_plots = not run_animations
    run_means_under_parameter_change_animation = False
    
    scale = "log"
    
    # Visualize Well Mixed
    if run_well_mixed:
        print("Plotting Well Mixed")
        visualize = DebtDeflationVisualization(filename_parameter_addon, show_plots)
        
        # Single companies and mean
        visualize.plot_companies(N_plot=4, scale=scale)
        visualize.plot_means(scale=scale)
        if run_means_under_parameter_change_animation: visualize.animate_mean_under_parameter_change()
        
        # # Size distributions
        visualize.final_time_size_dist()
        if run_animations: visualize.animate_size_distribution()

        # Inflation rate
        visualize.plot_inflation_rate()

        # # Values of all companies along x-axis
        visualize.final_time_values(scale=scale)
        if run_animations: visualize.animate_values(scale=scale)
        

    # Visualize 1d
    if run_1d:    
        print("Plotting 1d")
        visualize_1d = DebtDeflationVisualization(filename_parameter_addon_1d, show_plots)
        
        # Single companies and company mean
        visualize_1d.plot_companies(N_plot=4, scale=scale)
        visualize_1d.plot_means(scale=scale)
        if run_means_under_parameter_change_animation: visualize_1d.animate_mean_under_parameter_change()
    
        # Size distrubtions
        visualize_1d.final_time_size_dist()
        if run_animations: visualize_1d.animate_size_distribution()
        
        # Inflation rate
        visualize_1d.plot_inflation_rate()
        
        # Values of all companies along x-axis
        visualize_1d.final_time_values(scale=scale)
        if run_animations: visualize_1d.animate_values(scale=scale)
    
    if not show_plots: 
        plt.close()
        print("Finished plots!")
