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
        self.interest_rate = data_all[0, :, 4]
        self.interest_rate_PD_adjusted = data_all[1, :, 4]
        self.beta = data_all[:, :, 5]  
        self.did_not_take_loan = data_all[0, :, 6]  # Identical for all N companies on the 0-axis, so choose one
        self.first_derivative = data_all[0, :, 7]
        self.second_derivative = data_all[1, :, 7]
        
        # Derivates of data
        self.bank_money = self.bank_fortune - np.sum(self.debt, axis=0)
        self.time_steps = np.shape(self.production)[1]
        self.time_vals = np.arange(self.time_steps)
        self.N = np.shape(self.production)[0]
        

    def _count_consecutive_signs(self, arr) -> list:
        """Count consecutive positive/negative values in an array. Negative values are recorded as minus their count

        Args:
            arr (ndarray): Array of values 

        Returns:
            list: Counts of consecutive positive/negative values
        """
        result = []
        count = 1
        sign = np.sign(arr[0])
        
        for i in range(1, len(arr)):
            current_sign = np.sign(arr[i])
            if current_sign == sign:  # Consecutive of same values increase count
                count += 1
            else:  # Change of sign, record current count and reset
                result.append(count if sign > 0 else -count) 
                count = 1
                sign = current_sign
        
        # Append last count
        result.append(count if sign > 0 else -count)
        return result

    
    def plot_bank_fortune(self):
        fig, ax = plt.subplots()

        # Bank fortune 
        # To plot in log, plot negative and positive values separately, and plot abs(negative). 
        mask_neg_fortune = self.bank_fortune < 0
        mask_pos_fortune = self.bank_fortune > 0
        ax.plot(self.time_vals[mask_neg_fortune], np.abs(self.bank_fortune[mask_neg_fortune]), ".", color="red", label="abs(F<0)", alpha=0.9)
        ax.plot(self.time_vals[mask_pos_fortune], self.bank_fortune[mask_pos_fortune], ".", color="green", label="F>0", alpha=0.9)

        # Axis setup
        ax.set(ylabel="Log \$", title="Bank fortune", yscale="log")
        ax.grid()
        ax.legend(ncols=2, bbox_to_anchor=(0.5, 0.95), loc="lower center", fontsize=8)
        
        # Text Parameters, save and show
        self._add_parameters_text(ax) 
        self._save_fig(fig, name="bankfortune")
        if self.show_plots: plt.show()

        
    def plot_bank_fortune_components(self):
        fig, ax = plt.subplots()
        
        # Get total debt over time
        debt_summed = np.sum(self.debt, axis=0) 
        
        # Money
        mask_neg_money = self.bank_money < 0
        mask_pos_money = self.bank_money > 0
        ax.plot(self.time_vals[mask_neg_money], np.abs(self.bank_money[mask_neg_money]), ".", color="red", label="abs(M<0)", markersize=2)
        ax.plot(self.time_vals[mask_pos_money], self.bank_money[mask_pos_money], ".", color="green", label="M>0", markersize=2)

        # Debt
        ax.plot(self.time_vals, debt_summed, label="Debt", marker=".", markersize=2)
                
        # Axis setup
        ax.set(ylabel="Log \$", title="Bank fortune components", yscale="log")
        ax.grid()
        ax.legend(ncols=5, bbox_to_anchor=(0.5, 0.95), loc="lower center", fontsize=5)
        
        self._add_parameters_text(ax)
        self._save_fig(fig, name="bankfortune_components")
        if self.show_plots: plt.show()
        
        
    def plot_beta_evolution(self):
        # Preprocess
        beta_min, beta_max = self.beta.min(), self.beta.max()
        beta_mean = np.mean(self.beta, axis=0)
        
        # Create fig
        fig, (ax, ax1, ax2) = plt.subplots(nrows=3)
        
        # Plots
        # ax: Interest rate and adjusted interest rate
        ax.plot(self.time_vals, self.interest_rate, "-", label=r"$r$")
        ax.plot(self.time_vals, self.interest_rate_PD_adjusted, "--", label=r"$r$ adjusted")
        
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
        # Add parameters
        self._add_parameters_text(ax)
        # Save and show fig
        self._save_fig(fig, name="no_loan")
        if show_plots: plt.show()
            
    
    def plot_derivatives(self):
        """First column is log scale of first and second derivative, with abs(negative values) in red and positive in green.
        Second column show only the sign of the first and second derivative.
        """
        # Create figure and extract axis
        fig, ax_arr = plt.subplots(nrows=2, ncols=2, figsize=(16, 8))
        ax = ax_arr[0, 0]
        ax1 = ax_arr[0, 1]
        ax2 = ax_arr[1, 0]
        ax3 = ax_arr[1, 1]
                
        # Create masks for showing positive and negative values on log scale
        mask_neg_1st = self.first_derivative <= 0
        mask_pos_1st = self.first_derivative > 0
        mask_neg_2nd = self.second_derivative <= 0
        mask_pos_2nd = self.second_derivative > 0
        time_neg_1st = self.time_vals[mask_neg_1st]
        time_pos_1st = self.time_vals[mask_pos_1st]
        time_neg_2nd = self.time_vals[mask_neg_2nd]
        time_pos_2nd = self.time_vals[mask_pos_2nd]
        
        # Change any values of zero to a small value to avoid log(0) = -inf
        self.first_derivative[self.first_derivative == 0] = 1e-3
        self.second_derivative[self.second_derivative == 0] = 1e-3

        # Find where the sign changes
        # Identify sign change points for first and second derivatives
        sign_change_1st = np.diff(np.sign(self.first_derivative)) != 0
        sign_change_2nd = np.diff(np.sign(self.second_derivative)) != 0

        # Time values where sign changes occur
        time_change_1st = self.time_vals[:-1][sign_change_1st]  # :-1 because difference is one less than original
        time_change_2nd = self.time_vals[:-1][sign_change_2nd]

        # First column - Log scale values
        ax.plot(time_neg_1st, np.abs(self.first_derivative[mask_neg_1st]), ".", color="red", label="Negative", markersize=1)
        ax.plot(time_pos_1st, self.first_derivative[mask_pos_1st], ".", color="green", label="Positive", markersize=1) 
        ax1.plot(time_neg_2nd, np.abs(self.second_derivative[mask_neg_2nd]), ".", color="red", markersize=1, label="Negative")
        ax1.plot(time_pos_2nd, self.second_derivative[mask_pos_2nd], ".", color="green", markersize=1, label="Positive")
        
        # Second column - Sign values at +1 and -1
        ax2.plot(time_neg_1st, -np.ones_like(time_neg_1st), ".", color="red", markersize=2, label="Negative")
        ax2.plot(time_pos_1st, np.ones_like(time_pos_1st), ".", color="green", markersize=2, label="Positive")
        ax3.plot(time_neg_2nd, -np.ones_like(time_neg_2nd), ".", color="red", markersize=2, label="Negative") 
        ax3.plot(time_pos_2nd, np.ones_like(time_pos_2nd), ".", color="green", markersize=2, label="Positive")
        
        # Add markers at sign change points
        ax2.plot(time_change_1st, np.zeros_like(time_change_1st), ".", color="blue", markersize=1, label="Sign change")
        ax3.plot(time_change_2nd, np.zeros_like(time_change_2nd), ".", color="blue", markersize=1, label="Sign change")
        
        # Axis setup
        ax.set(ylabel="Log Derivative", title="First derivative", yscale="log")
        ax1.set(xlabel="Time", title="Second derivative", yscale="log")
        ax2.set(ylabel="Sign of derivative", title="Sign of first derivative", yticks=[-1, 1], ylim=(-1.5, 1.5), xlabel="Time")
        ax3.set(title="Sign of second derivative", yticks=[-1, 1], ylim=(-1.5, 1.5), xlabel="Time")
        
        # Legend
        ax.legend(ncols=2, bbox_to_anchor=(0.5, 0.8), loc="lower center")
        ax1.legend(ncols=2, bbox_to_anchor=(0.5, 0.8), loc="lower center")
        ax2.legend(ncols=3, bbox_to_anchor=(0.5, 0.9), loc="lower center")
        ax3.legend(ncols=3, bbox_to_anchor=(0.5, 0.9), loc="lower center")
        
        # Grid
        ax.grid()
        ax1.grid() 
        ax2.grid() 
        ax3.grid()
        # Add parameters
        self._add_parameters_text(ax, x=0.05)
        # Save and show fig
        self._save_fig(fig, name="derivatives")
        if self.show_plots: plt.show()


    def plot_consecutive_counts(self):
        """Plot consecutive counts of positive and negative values for first and second derivatives."""
        # Calculate consecutive counts
        consecutive_counts_1st = self._count_consecutive_signs(self.first_derivative)
        consecutive_counts_2nd = self._count_consecutive_signs(self.second_derivative)
        
        # Rescale time to match counts by taking the cumulative sum of the counts with and extra zero added at the start. Therefore, has to exclude the last element
        time_vals_1st = np.cumsum(np.concatenate((np.abs(consecutive_counts_1st), [0])))[:-1]
        time_vals_2nd = np.cumsum(np.concatenate((np.abs(consecutive_counts_2nd), [0])))[:-1]
        
        # Create figure and extract axis
        fig, (ax, ax1) = plt.subplots(nrows=2)
        
        # Plot consecutive counts for first derivative
        ax.plot(time_vals_1st, consecutive_counts_1st, "--.", markersize=4)        
        ax1.plot(time_vals_2nd, consecutive_counts_2nd, "--.", markersize=4)

        ax.set(ylabel="Counts", title="First Derivative")
        ax1.set(xlabel="Time", ylabel="Counts", title="Second Derivative")

        # Grid
        ax.grid()
        ax1.grid()
        
        # Save and show plot
        self._save_fig(fig, name="consecutive_counts")
        if show_plots: plt.show()
        
    
    def plot_(self):
        # Load data
        filename = Path.joinpath(self.dir_path_output, "repeated_" + self.filename + ".npy")
        data = np.load(filename)
        beta_arr = data[:, :, 0]  # Shape (companies, repeats, variable)
        interest_rate_arr = data[:, :, 1]
        interest_rate_PD_adjusted_arr = data[:, :, 2]
        production_arr = data[:, :, 3]
        # Take mean over companies
        beta_mean = np.mean(beta_arr, axis=0)
        interest_rate_mean = np.mean(interest_rate_arr, axis=0)
        interest_rate_PD_adjusted_mean = np.mean(interest_rate_PD_adjusted_arr, axis=0)
        production_mean = np.mean(production_arr, axis=0)
        
        # Std over companies
        beta_std = np.std(beta_arr, axis=0, ddof=1)
        interest_rate_std = np.std(interest_rate_arr, axis=0, ddof=1)
        interest_rate_PD_adjusted_std = np.std(interest_rate_PD_adjusted_arr, axis=0, ddof=1)
        production_std = np.std(production_arr, axis=0, ddof=1)
        
        # Create figure
        fig, (ax, ax1) = plt.subplots(nrows=2)
        
        # Plots
        ax.errorbar(beta_mean, production_mean, xerr=beta_std, yerr=production_std, fmt="o")
        ax1.errorbar(beta_mean, interest_rate_mean, xerr=beta_std, yerr=interest_rate_std, fmt="o")
        
        # Axis setup
        ax.set(ylabel="Production", title=r"Production vs $\beta$")
        ax1.set(ylabel="Interest rate", xlabel=r"$\beta$", title=r"Interest rate vs $\beta$")
        
        # Parameters
        self._display_parameters(ax)
        
        # Save and show fig        
        self._save_fig(fig, name="variable_dependence")
        if self.show_plots: plt.show()
        


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
        visualize.plot_derivatives()
        visualize.plot_consecutive_counts()
        
        if animate:
            visualize.animate_values(scale)
            # visualize.animate_beta_evolution()

    
    plt.close()
    print("Finished plotting")