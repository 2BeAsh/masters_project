import numpy as np
import general_functions
from tqdm import tqdm
from time import time
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.animation as animation
import functools


class DebtDeflationVisualization():
    def __init__(self, filename):
        # Local paths for saving files.
        self.dir_path = "code/"
        self.dir_path_output = self.dir_path + "output/"
        self.dir_path_image = self.dir_path + "image/"

        self.filename = filename


    def display_parameters(self):
        # Split at underscores "_"
        filename_list = filename.split("_")
        
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
        company_value, debt, money = self._load_data()
        
        # Averages
        company_value_mean = np.mean(company_value, axis=0)
        debt_mean = np.mean(debt, axis=0)
        money_mean = np.mean(money, axis=0)
        time_values = np.arange(0, self.time_steps)
        
        fig, ax = plt.subplots()
        ax.plot(time_values, company_value_mean, label="Company production")
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
        company_value, debt, money = self._load_data()
        company_value_plot = company_value[:N_plot, :].T
        debt_plot = debt[: N_plot, :].T
        money_plot = money[: N_plot, :].T
        time_values = np.arange(0, self.time_steps)
        
        # Plot averages single axis
        fig, (ax0, ax1, ax2) = plt.subplots(nrows=3)
        mask = slice(0, self.time_steps)
        ax0.plot(time_values[mask], company_value_plot[mask],)
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


    def animate_size_distribution(self):
        time_i = time()
        # Load data and create time values array
        company_value, debt, money = self._load_data(self.filename)
        # Bin data
        Nbins = int(np.sqrt(self.time_steps))
        bin_edges = np.linspace(company_value.min(), company_value.max(), Nbins)
        
        fig, ax = plt.subplots()
        # n, _ = np.histogram(company_value[:, 0], bin_edges)  
        _, _, bar_container = ax.hist(company_value[:, 0], bin_edges)  # Initial histogram 
        ax.set(xlim=(bin_edges[0], bin_edges[-1]), title="Time = 0")
        # Text
        display_parameters_str = self.display_parameters()
        # display_parameters_str = (r"$N_{agent} = $" + str(self.N) 
        #                           + "\n" + r"Interest $=$ " + str(self.r) 
        #                           + "\n" + r"$\alpha = $ " + str(self.alpha)
        #                           + "\n" + r"$\beta = $ " + str(self.beta))
        ax.text(x=0.01, y=0.9, s=display_parameters_str, transform=ax.transAxes, horizontalalignment='left', verticalalignment='center', fontsize=9)
  
        def animate(i, bar_container):
            data = company_value[:, i]
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
    # visualize = DebtDeflationVisualization(number_of_companies=N_agents, 
    #                               money_to_production_efficiency=money_to_production_efficiency, 
    #                               loan_probability=loan_probability, 
    #                               interest_rate=interest, 
    #                               buy_fraction=buy_fraction, 
    #                               equilibrium_distance_fraction=equilibrium_distance_fraction, 
    #                               time_steps=time_steps)
    
    filename = "T1000_N100_r1_alpha0.05_beta0.0_sigma0.9_epsilon0.1"
    visualize = DebtDeflationVisualization(filename)
    
    visualize.plot_companies(N_plot=4)
    visualize.plot_means()
    
    # visualize.animate_size_distribution()