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
    def __init__(self, filename):
        # Local paths for saving files.
        self.dir_path = "code/"
        self.dir_path_output = self.dir_path + "output/"
        self.dir_path_image = self.dir_path + "image/"

        self.filename = filename


    def display_parameters(self):
        # Split at underscores "_"
        filename_list = self.filename.split("_")
        
        # Combine strings and add newlines
        filename_comb = ""
        for arg_name in filename_list:
            filename_comb += arg_name + "\n"
        return filename_comb
        # display_parameters_str = (r"$N_{agent} = $" + str(self.N) 
        #                           + "\n" + r"Interest $=$ " + str(self.r) 
        #                           + "\n" + r"$\alpha = $ " + str(self.alpha)
        #                           + "\n" + r"$\beta = $ " + str(self.beta))


    def _load_data(self) -> None:
        data_all = np.load(self.dir_path_output + self.filename + ".npy")
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
        display_parameters_str = self.display_parameters()
        ax.text(x=0.1, y=0.8, s=display_parameters_str, transform=ax.transAxes, horizontalalignment='left', verticalalignment='center', fontsize=9)

        figname = self.dir_path_image + f"means_" + self.filename + ".png"
        plt.savefig(figname)
        plt.show()


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
        display_parameters_str = self.display_parameters()
        ax0.text(x=0.1, y=0.8, s=display_parameters_str, transform=ax0.transAxes, horizontalalignment='left', verticalalignment='center', fontsize=9)
        fig.suptitle(f"First {N_plot} companies", fontsize=15, fontstyle="italic")
        # Save figure
        figname = self.dir_path_image + f"single_companies_" + self.filename + ".png"
        plt.savefig(figname)
        plt.show()


    def final_time_size_dist(self):
        production, debt, money = self._load_data()
        Nbins = int(np.sqrt(self.time_steps))
        bin_edges = np.linspace(production.min(), production.max(), Nbins)
        
        production_final = production[:, -1]
        fig, ax = plt.subplots()
        ax.hist(production_final, bins=Nbins)
        ax.set(xlabel="Production", title="Final time production distribution", ylabel="Frequency")
        plt.show()
        

    def animate_size_distribution(self):
        time_i = time()
        # Load data and create time values array
        production, debt, money = self._load_data()
        # Bin data
        Nbins = int(np.sqrt(self.time_steps))
        bin_edges = np.linspace(production.min(), production.max(), Nbins)
        
        fig, ax = plt.subplots()
        # n, _ = np.histogram(production[:, 0], bin_edges)  
        _, _, bar_container = ax.hist(production[:, 0], bin_edges)  # Initial histogram 
        ax.set(xlim=(bin_edges[0], bin_edges[-1]), title="Time = 0")
        # Text
        display_parameters_str = self.display_parameters()
        # display_parameters_str = (r"$N_{agent} = $" + str(self.N) 
        #                           + "\n" + r"Interest $=$ " + str(self.r) 
        #                           + "\n" + r"$\alpha = $ " + str(self.alpha)
        #                           + "\n" + r"$\beta = $ " + str(self.beta))
        ax.text(x=0.01, y=0.9, s=display_parameters_str, transform=ax.transAxes, horizontalalignment='left', verticalalignment='center', fontsize=9)
  
        def animate(i, bar_container):
            data = production[:, i]
            n, _ = np.histogram(data, bin_edges)
            for count, rect in zip(n, bar_container.patches):
                rect.set_height(count)
            
            ax.set_title(f"Time = {i}")
            return bar_container.patches
        
        anim = functools.partial(animate, bar_container=bar_container)
        ani = animation.FuncAnimation(fig, anim, frames=self.time_steps, interval=1)
        
        time_create_ani = time()
        
        animation_name = self.dir_path_image + "size_distribution_animation_" + self.filename + ".mp4"
        ani.save(animation_name, fps=30)
        
        time_save_ani = time()
        print("Time creating animation: \t", time_create_ani - time_i)
        print("Time saving animation: \t", time_save_ani - time_create_ani)
        

if __name__ == "__main__":      
    run_well_mixed = True
    run_1d = True
    
    
    # Visualize Well Mixed
    if run_well_mixed:
        #filename = "Steps1000_Companies100_Interest1_Efficiency0.05_LoanProb0.0_BuyFraction1_EquilibriumStep0.01"
        filename = filename_parameter_addon
        visualize = DebtDeflationVisualization(filename)
        
        visualize.plot_companies(N_plot=4)
        visualize.plot_means()
        
        # visualize.animate_size_distribution()
        visualize.final_time_size_dist()


    # Visualize 1d
    if run_1d:    
        visualize_1d = DebtDeflationVisualization(filename_parameter_addon_1d)
        
        visualize_1d.plot_companies(N_plot=4)
        visualize_1d.plot_means()
    
        # visualize.animate_size_distribution()
        visualize_1d.final_time_size_dist()