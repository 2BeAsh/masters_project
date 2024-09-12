import numpy as np
from time import time
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.animation as animation
import functools
from pathlib import Path
# My files
import general_functions  # Own library, here only used for matplotlib styling
from bank_well_mixed import filename_parameter_addon
from debt_deflation_plots import DebtDeflationVisualization
from tqdm import tqdm


class BankVisualization(DebtDeflationVisualization):
    def __init__(self, filename, show_plots):
        # Get master methods and variables
        super().__init__(filename, show_plots)
        
        # Image path
        filename_split = filename.split("_")
        image_sufix = filename_split[-1] + "/"
        self.dir_path_image = Path.joinpath(self.dir_path, "image_bank", image_sufix)
        
        # Load data
        filename = Path.joinpath(self.dir_path_output, self.filename + ".npy")
        data_all = np.load(filename)
        self.production = data_all[:, :, 0]
        self.debt = data_all[:, :, 1]
        self.money = data_all[:, :, 2] 
        self.bank_fortune = data_all[0, :, 3]  # Identical for all N companies on the 0-axis, so choose one
        self.interest_rate = data_all[0, :, 4]  # Identical for all N companies on the 0-axis, so choose one
        self.beta = data_all[:, :, 5]  
        self.did_not_take_loan = data_all[0, :, 6]  # Identical for all N companies on the 0-axis, so choose one
        
        # Derivates of data
        self.bank_money = self.bank_fortune - np.sum(self.debt, axis=0)
        self.time_steps = np.shape(self.production)[1]
        self.time_vals = np.arange(self.time_steps)
        self.N = np.shape(self.production)[0]
        
    
    def plot_bank_fortune(self):
        fig, ax = plt.subplots()
        
        # Get total debt over time
        debt_summed = np.sum(self.debt, axis=0) 
        
        # Fortune - Plot the two terms of fortune i.e. bank money and debt
        # Money
        mask_neg_money = self.bank_money < 0
        mask_pos_money = self.bank_money > 0
        ax.plot(self.time_vals[mask_neg_money], np.abs(self.bank_money[mask_neg_money]), ".", color="red", label="Negative Money", markersize=2)
        ax.plot(self.time_vals[mask_pos_money], self.bank_money[mask_pos_money], ".", color="green", label="Positive Money", markersize=2)

        # Debt
        ax.plot(self.time_vals, debt_summed, label="Debt", marker=".", markersize=2)
        
        # Bank fortune 
        ax.plot(self.time_vals, self.bank_fortune, "--", color="black", label="Abs Bank fortune", markersize=1, alpha=0.7)
        
        # Axis setup
        ax.set(ylabel="Log \$", title="Bank fortune and its components", yscale="log")
        ax.grid()
        ax.legend()
        
        self._add_parameters_text(ax)
        self._save_fig(fig, name="bankfortune")
        if self.show_plots: plt.show()
        
        
    def plot_beta_evolution(self):
        # Preprocess
        beta_min, beta_max = self.beta.min(), self.beta.max()
        beta_mean = np.mean(self.beta, axis=0)
        
        # Create fig
        fig, (ax, ax1, ax2) = plt.subplots(nrows=3)
        
        # Plots
        ax.plot(self.time_vals, self.interest_rate)
        ax1.plot(self.time_vals, beta_mean)
        im = ax2.imshow(self.beta, cmap="hot", vmin=beta_min, vmax=beta_max, aspect="auto", origin="lower")
        
        # Colorbar
        fig.colorbar(im)
        
        # Axis setup
        ax.set(ylabel="Interest rate", xlabel="Time", title="Interest rate", xlim=(0, self.time_steps))
        ax1.set(ylabel=r"Mean $\beta$", xlabel="Time", title=r"Company Mean $\beta$", xlim=(0, self.time_steps))
        ax2.set(ylabel="Companies", xlabel="Time", title=r"$\beta$ evolution")
        
        ax.grid()
        ax1.grid()
        # Parameters
        self._add_parameters_text(ax)
        # Save and show fig
        self._save_fig(fig, name="beta_evolution")
        if self.show_plots: plt.show()


    def animate_beta_evolution(self):
        time_i = time()
        
        # Preprocess
        beta_min, beta_max = self.beta.min(), self.beta.max()
        beta_mean = np.mean(self.beta, axis=0)
        beta_mean_min = beta_mean.min()
        beta_mean_max = beta_mean.max()
        interest_min = self.interest_rate.min()
        interest_max = self.interest_rate.max() 
        tmin = self.time_vals[0]
        tmax = self.time_vals[-1]
        
        beta_arr_empty = -1 * np.ones_like(self.beta)
        
        # Create fig
        fig, (ax, ax1, ax2) = plt.subplots(nrows=3)
        
        # Initial lines Plots
        line1, = ax.plot([], [], "-")  
        line2, = ax1.plot([], [], ".", markersize=2)
        im = ax2.imshow(beta_arr_empty, cmap="hot", vmin=beta_min, vmax=beta_max, aspect="auto", origin="lower")
                
        # Colorbar
        fig.colorbar(im)
        
        # Axis setup
        ax.set(ylabel="Interest rate", xlabel="Time", title="Interest rate",
               xlim=(tmin, tmax), ylim=(interest_min, interest_max))
        ax1.set(ylabel=r"Mean $\beta$", xlabel="Time", title=r"Company Mean $\beta$",
                xlim=(tmin, tmax), ylim=(beta_mean_min, beta_mean_max))
        ax2.set(ylabel="Companies", xlabel="Time", title=r"$\beta$ evolution")
        
        ax.grid()
        ax1.grid()

        # def update(frame):
        #     line1.set_data(time_vals[:frame], interest_rate[:frame])
        #     line2.set_data(time_vals[:frame], beta_mean[:frame])
        #     beta_arr_empty[:, :frame] = beta[:, :frame]
        #     im.set_array(beta_arr_empty)
        #     return line1, line2, im
        
        def update(frame):
            line1.set_data(self.time_vals[:frame], self.interest_rate[:frame])
            line2.set_data(self.time_vals[:frame], beta_mean[:frame])
            beta_arr_empty[:, :frame] = self.beta[:, :frame]
            im.set_array(beta_arr_empty)
            pbar.update(1)  # Update progress bar
            fig.suptitle(f"TIme = {frame}")
            return line1, line2, im
        
        with tqdm(total=self.time_steps) as pbar:
            ani = animation.FuncAnimation(fig, update, frames=self.time_steps, interval=100, blit=True)
        time_create = time()
        self._save_anim(ani, name="beta_evolution")
        time_save = time()
        
        print("Time create anim = ", (time_create - time_i) / 60, " minutes")
        print("Time save = anim = ", (time_save - time_create) / 60, " minutes")


    def plot_did_not_take_loan(self):
        fraction_did_not_take_loan = self.did_not_take_loan[1:] / self.N  # Remove first entry because no data
        # Create figure
        fig, ax = plt.subplots()
        ax.plot(self.time_vals[1:], fraction_did_not_take_loan)        
        ax.set(xlabel="Time", ylabel=r"$n/N$", title=f"Fraction of companies that did not take a loan, N={self.N}")
        
        if show_plots: plt.show()
        

if __name__ == "__main__": 
    run_wm = True  # Well mixed
    show_plots = True
    animate = False
    scale = "log"
        
    if run_wm:
        print("Plotting Well Mixed")
        visualize = BankVisualization(filename_parameter_addon, show_plots)
        
        visualize.plot_companies(N_plot=4, scale=scale)
        visualize.plot_means(scale)
        visualize.final_time_values(scale)
        visualize.final_time_size_dist()
        visualize.plot_bank_fortune()
        visualize.plot_beta_evolution()
        visualize.plot_did_not_take_loan()
        
        if animate:
            visualize.animate_values(scale)
            # visualize.animate_beta_evolution()

    
    plt.close()
    print("Finished plotting")