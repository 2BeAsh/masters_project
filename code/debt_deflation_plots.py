import numpy as np
import general_functions
from tqdm import tqdm
from time import time
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.animation as animation
import functools
from debt_deflation_well_mixed import filename_parameter_addon
from debt_deflation_1d import filename_parameter_addon_1d


class DebtDeflationVisualization():
    def __init__(self, filename, show_plots=False):
        # Local paths for saving files.
        self.filename = filename
        self.dir_path = "code/"
        self.dir_path_output = self.dir_path + "output/"
        
        # Get image path from filename
        filename_split = filename.split("_")
        image_sufix = filename_split[-1] + "/"
        self.dir_path_image = self.dir_path + "image/" + image_sufix

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
        if parameter_change:
            filename = self.dir_path_output + self.filename + "_parameter_change" + ".npy"
            data_all = np.load(filename)
            production = data_all[:, :, 0]
            debt = data_all[:, :, 1]
            money = data_all[:, :, 2] 
            
            r_vals = data_all[:, :, 3]
            r_vals_true = r_vals[:, 0]
            
            self.time_steps = np.shape(production)[1]
            return production, debt, money, r_vals_true
        
        else:
            filename = self.dir_path_output + self.filename + ".npy"
            data_all = np.load(filename)
            production = data_all[:, :, 0]
            debt = data_all[:, :, 1]
            money = data_all[:, :, 2] 
            self.time_steps = np.shape(production)[1]
            return production, debt, money
    
    
    def plot_means(self):
        # Load data and create time values array
        production, debt, money = self._load_data()
        
        # Averages
        production_mean = np.mean(production, axis=0)
        debt_mean = np.mean(debt, axis=0)
        money_mean = np.mean(money, axis=0)
        time_values = np.arange(0, self.time_steps)
        
        fig, ax = plt.subplots()
        ax.plot(time_values, production_mean, label="Company production")
        ax.plot(time_values, debt_mean, label="Debt")
        ax.plot(time_values, money_mean, "--", label="Money")
        
        ax.set(xlabel="Time", ylabel="$", title="Mean values")
        # Figure setup
        legend_elements = [Line2D([], [], color="rebeccapurple", label="Production"),
                           Line2D([], [], color="firebrick", label="Debt"),
                           Line2D([], [], color="black", label="Money"),]
        ax.legend(handles=legend_elements, ncols=3, bbox_to_anchor=(0.5, 0.9), loc="lower center")

        # Display parameters
        self._add_parameters_text(ax)

        # Save figure
        figname = self.dir_path_image + f"means_" + self.filename + ".png"
        plt.savefig(figname)
        if self.show_plots: plt.show()


    def plot_companies(self, N_plot):
        """Plot averages.
        """
        # Load data and create time values array
        production, debt, money = self._load_data()
        production_plot = production[:N_plot, :].T
        debt_plot = debt[: N_plot, :].T
        money_plot = money[: N_plot, :].T
        time_values = np.arange(0, self.time_steps)
        
        # Plot averages single axis
        fig, (ax0, ax1, ax2) = plt.subplots(nrows=3)
        mask = slice(0, self.time_steps)
        ax0.plot(time_values[mask], production_plot[mask],)
        ax1.plot(time_values[mask], debt_plot[mask])
        ax2.plot(time_values[mask], money_plot[mask] )

        # Figure setup
        ax0.set(ylabel="$", title="Production")# yscale="log")
        ax1.set(ylabel="$", title="Debt",)# yscale="log")
        ax2.set(ylabel="$", xlabel="Time", title="Money")
        
        # Display parameter values
        self._add_parameters_text(ax0)
        
        fig.suptitle(f"First {N_plot} companies", fontsize=15, fontstyle="italic")
        # Save figure
        figname = self.dir_path_image + f"single_companies_" + self.filename + ".png"
        plt.savefig(figname)
        if self.show_plots: plt.show()


    def final_time_size_dist(self):
        production, debt, money = self._load_data()
        Nbins = int(np.sqrt(self.time_steps))
        bin_edges = np.linspace(production.min(), production.max(), Nbins)
        
        production_final = production[:, -1]
        fig, ax = plt.subplots()
        n, _, _ = ax.hist(production_final, bins=Nbins)
        ax.set(xlabel="Production", title="Final time production distribution", ylabel="Frequency")

        # Parameters text
        self._add_parameters_text(ax)

        # Save figure
        figname = self.dir_path_image + f"final_time_dist_" + self.filename + ".png"
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
        figname = self.dir_path_image + f"final_time_values" + self.filename + ".png"
        plt.savefig(figname)
        if self.show_plots: plt.show()


    def animate_size_distribution(self):
        time_i = time()
        # Load data and create time values array
        production, debt, money = self._load_data()
        
        # Bin data
        Nbins = int(np.sqrt(self.time_steps))
        bin_edges = np.linspace(production.min(), production.max(), Nbins)
        
        # Figure setup        
        fig, ax = plt.subplots()
        _, _, bar_container = ax.hist(production[:, 0], bin_edges)  # Initial histogram 
        ax.set(xlim=(bin_edges[0], bin_edges[-1]), title="Time = 0")

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
        animation_name = self.dir_path_image + "size_distribution_animation_" + self.filename + ".mp4"
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
        
        # If using logscale, set minimum value as log(0) = -infty
        y_min = np.min([production, debt, money])
        if scale == "log":
            y_min = 1e-3
            money = np.maximum(money, y_min)
            debt = np.maximum(debt, y_min)
        
        # Initial image and plot setup
        if on_same_row:
            # Find ylim
            y_max = np.max([production, debt, money])
            ylim = (y_min, y_max)
            
            fig, ax = plt.subplots()        
            line_prod = ax.plot(x_vals, production[:, 0], ".", label="Production")[0]
            line_debt = ax.plot(x_vals, debt[:, 0], "x", label="Debt", alpha=0.9)[0]
            line_money = ax.plot(x_vals, money[:, 0], "*", label="Money", alpha=0.8)[0]
            ax.legend(ncols=3, bbox_to_anchor=(0.5, 0.9), loc="lower center")
            ax.set_yscale(scale)
            ax.set(xlim=(0, N), ylim=ylim, xlabel="Company Number", ylabel="$")
            ax.grid()
            # Parameters text
            self._add_parameters_text(ax)
            
        else:
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
            ax_p.set(xlim=(0, N), ylim=(np.min(production), np.max(production)), ylabel="Production value")
            ax_d.set(xlim=(0, N), ylim=(np.min(debt), np.max(debt)), ylabel="Debt")
            ax_m.set(xlim=(0, N), ylim=(np.min(money), np.max(money)), xlabel="Company Number", ylabel="Money")
            # Grid
            ax_p.grid()
            ax_d.grid()
            ax_m.grid()
            # Ticks
            ax_p.set_xticks(ticks=x_vals, labels=x_vals, fontsize=3)
            ax_d.set_xticks(ticks=x_vals, labels=x_vals, fontsize=3)
            ax_m.set_xticks(ticks=x_vals, labels=x_vals, fontsize=3)
            # Parameters text
            self._add_parameters_text(ax_p)

        
        # Line update function
        def animate(i):
            line_prod.set_ydata(production[:, i])
            line_debt.set_ydata(debt[:, i])
            line_money.set_ydata(money[:, i])
            fig.suptitle(f"Time = {i}")
        
        # Create animation and save it
        anim = animation.FuncAnimation(fig, animate, interval=1, frames=self.time_steps)
        
        time_create_anim = time()  # Record time
        animation_name = self.dir_path_image + "value_animation_" + f"{scale}_same_row{on_same_row}_" + self.filename + ".mp4"
        anim.save(animation_name, fps=30)
        
        # Display times
        time_save_anim = time()
        print("Time creating animation: \t", time_create_anim - time_i)
        print("Time saving animation: \t", time_save_anim - time_create_anim)
        
        
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
               xlim=(time_values[0], time_values[-1]), )
            #    ylim=(ymin, ymax))
        
        # Legend
        legend_elements = [Line2D([], [], color="rebeccapurple", label="Production"),
                           Line2D([], [], color="firebrick", label="Debt"),
                           Line2D([], [], color="black", label="Money"),]
        ax.legend(handles=legend_elements, ncols=3, bbox_to_anchor=(0.5, 0.9), loc="lower center")

        # Display parameters
        # self._add_parameters_text(ax)


        # Create initial lines
        line_p = ax.plot(time_values, production_means[0, :])[0]
        line_d = ax.plot(time_values, debt_means[0, :])[0]
        line_m = ax.plot(time_values, money_means[0, :], "--")[0]
        
        # Update function
        def animate(i):
            line_p.set_ydata(production_means[i, :])
            line_d.set_ydata(debt_means[i, :])
            line_m.set_ydata(money_means[i, :])
            
            # Title. r value, time_steps (and repeats?)
            r_val_i = r_vals[i]
            ax.set_title(label=f"r = {r_val_i}, steps = {i}, {i % N_repeats}/{N_repeats}")
            
            # y limits. Only update ylim when new r value
            if i % N_repeats == 0:
                # Find min and max y value of the next N_repeats values
                ymin = np.min([production_means[i: i + N_repeats, :], debt_means[i: i + N_repeats, :], money_means[i: i + N_repeats, :]])
                ymax = np.max([production_means[i: i + N_repeats, :], debt_means[i: i + N_repeats, :], money_means[i: i + N_repeats, :]])
                ax.set_ylim(ymin, ymax)
            ax.set_xlim(time_values[0], time_values[-1])

        # Create animation and save it
        anim = animation.FuncAnimation(fig, animate, interval=1, frames=np.shape(production_means)[0])
        
        time_create_anim = time()  # Record time
        animation_name = self.dir_path_image + "parameter_change_animation_" + self.filename + ".mp4"
        anim.save(animation_name, fps=2)
        
        # Display times
        time_save_anim = time()
        print("Parameter change animation:")
        print("Time creating animation: \t", time_create_anim - time_i)
        print("Time saving animation: \t", time_save_anim - time_create_anim)
            

if __name__ == "__main__":      
    run_well_mixed = True
    run_1d = False
    show_plots = False
    run_animations = True
    
    
    # Visualize Well Mixed
    if run_well_mixed:
        #filename = "Steps1000_Companies100_Interest1_Efficiency0.05_LoanProb0.0_BuyFraction1_EquilibriumStep0.01"
        filename = filename_parameter_addon
        visualize = DebtDeflationVisualization(filename, show_plots)
        
        # Single companies and mean
        # visualize.plot_companies(N_plot=4)
        # visualize.plot_means()
        if run_animations: visualize.animate_mean_under_parameter_change()
        
        # # Size distributions
        # visualize.final_time_size_dist()
        # if run_animations: visualize.animate_size_distribution()

        # # Values of all companies along x-axis
        # visualize.final_time_values(scale="linear")
        # if run_animations: visualize.animate_values()
        
        

    # Visualize 1d
    if run_1d:    
        visualize_1d = DebtDeflationVisualization(filename_parameter_addon_1d, show_plots)
        
        # Single companies and company mean
        visualize_1d.plot_companies(N_plot=4)
        visualize_1d.plot_means()
    
        # Size distrubtions
        visualize_1d.final_time_size_dist()
        if run_animations: visualize_1d.animate_size_distribution()
        
        # Values of all companies along x-axis
        visualize_1d.final_time_values()
        if run_animations: visualize_1d.animate_values(scale="log", on_same_row=False)
    
    if not show_plots: print("Finished plots!")
