import general_functions 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib.colors import PowerNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import functools
import numpy as np
import scipy.optimize
import scipy.stats
from scipy.ndimage import uniform_filter1d
from run import dir_path_image
from postprocess import PostProcess


class PlotMaster(general_functions.PlotMethods, PostProcess):
    def __init__(self, data_group_name, skip_values=0, show_plots=True, add_parameter_text_to_plot=True, save_figure=True):
        general_functions.PlotMethods.__init__(self, data_group_name)
        PostProcess.__init__(self, data_group_name)
        
        # Allow for the option of specifying a group name. Otherwise the existing group name is used.
        self.group_name = data_group_name
        self.show_plots = show_plots
        self.save_figure = save_figure
        self.add_parameter_text_to_plot = add_parameter_text_to_plot
        self.dir_path_image = dir_path_image
        self.loaded_groups = {}
        
        self.skip_values = skip_values  
        
        # Colours
        self._set_colours()
    
    
    def _set_colours(self):
        """Set colours to salary, debt, interest rate etc
        """
        self.colours = {
            "salary": general_functions.list_of_colors[0],
            "debt": general_functions.list_of_colors[1],
            "interest_rate": general_functions.list_of_colors[2],
            "workers": general_functions.list_of_colors[3],
            "mutations": general_functions.list_of_colors[4],
            "bankruptcy": "red",
            "time": general_functions.list_of_colors[5],
            "diversity": general_functions.list_of_colors[6],
            "mu": general_functions.list_of_colors[7],
            "capital": general_functions.list_of_colors[8],
        }
    
    
    def _get_data(self, gname):
        """Redefine the _get_data method from PostProcess to include the xlim attribute.

        Args:
            gname (_type_): _description_
        """
        # Get the parent's method
        super()._get_data(gname)
        
        
        if (self.s != None).all():
            self.xlim = (self.skip_values, self.time_steps)


    def plot_salary(self, show_spread=False):
        """Plot the mean salary and fraction who went bankrupt on twinx. Plot the spread (std/mean) on a subplot below it."""
        self._get_data(self.group_name)
        mean_salary = self.s.mean(axis=0)[self.skip_values:]
        median_salary = np.median(self.s, axis=0)[self.skip_values:]
        fraction_bankrupt = (self.went_bankrupt[self.skip_values:] / self.N)
        spread = (self.s.std(axis=0)[self.skip_values:] / mean_salary)
        time_values = np.arange(self.skip_values, self.time_steps)
        
        # Create figure
        nrows = 1 if not show_spread else 2
        fig, ax0 = plt.subplots(nrows=nrows, figsize=(10, 8))
        
        if show_spread:
            ax0, ax1 = ax0
        
        # ax0 - Salary and fraction who went bankrupt
        c0 = self.colours["salary"]
        c1 = self.colours["bankruptcy"]
        
        # Bankruptcy
        ax0_twin = ax0.twinx()
        ax0_twin.plot(time_values, fraction_bankrupt, color=c1, label="Fraction bankrupt", alpha=0.3)
        ax0_twin.set_ylabel("Fraction bankrupt", color=c1)
        ax0_twin.tick_params(axis='y', labelcolor=c1)

        # Mean and median salary
        ax0.plot(time_values, mean_salary, label="Mean salary", c=c0, alpha=1)
        ax0.plot(time_values, median_salary, label="Median salary", c="black", alpha=0.7, ls="dotted")
        ax0.set(xlim=self.xlim, ylabel="Price", yscale="linear", title="Mean salary and bankruptcies")
        ax0.set_ylabel("Price", color=c0)
        ax0.tick_params(axis='y', labelcolor=c0)
        ax0.grid()
        self._add_legend(ax0, ncols=3, x=0.5, y=0.95)
        
        if show_spread:
            # ax1 - Spread
            ax1.plot(time_values, spread, label="Spread")
            ax1.set(xlabel="Time", xlim=self.xlim, ylabel="Spread", title="Spread (std/mean)")
            ax1.grid()
        
            # Plot the peaks as vertical lines on ax0 and ax1
            if np.any(self.peak_idx != None):
                for peak in self.peak_idx:
                    ax0.axvline(x=peak, ls="--", c="grey", alpha=0.7)
                    ax1.axvline(x=peak, ls="--", c="grey", alpha=0.7)
        
        self._text_save_show(fig, ax0, "salary", xtext=0.05, ytext=0.95, fontsize=6)
        
        
    def plot_single_companies(self, N_plot):
        """Plot the salary and debt of the first N_plot companies"""
        # Get data and remove skip_values
        self._get_data(self.group_name)
        
        # Get first N_plot companies
        s = self.s[:N_plot, self.skip_values:]
        d = self.d[:N_plot, self.skip_values:]
        w = self.w[:N_plot, self.skip_values:]
        time_values = np.arange(self.skip_values, self.time_steps)
                
        # Create figure
        fig, (ax_s, ax_d, ax_w) = plt.subplots(nrows=3, figsize=(10, 8))
        
        # ax_s - salary
        ylim = (0.99e-2, np.max(s)*1.01)
        ax_s.plot(time_values, s.T)
        ax_s.set(title=f"Salary and debt of first {N_plot} companies", yscale="log", ylim=ylim)
        ax_s.set_ylabel("Log salary")
        # ax_s.tick_params(axis='y', labelcolor=c0)
        ax_s.grid()
        
        # ax_d - debt
        # ax_d = ax_s.twinx()
        ax_d.plot(time_values, d.T,)
        ax_d.set(xlabel="Time", ylabel="Log Debt", yscale="symlog")
        ax_d.set_ylabel("Debt")
        # ax_d.tick_params(axis='y', labelcolor=c1)
        ax_d.grid()

        # Plot bankruptcies for the first company on the debt subplot
        idx_bankrupt = self.went_bankrupt_idx[0, self.skip_values:]
        time_bankrupt = time_values[idx_bankrupt]
        d_bankrupt = d[0, idx_bankrupt]
        s_bankrupt = s[0, idx_bankrupt]
        ax_d.scatter(time_bankrupt, d_bankrupt, c=self.colours["bankruptcy"], marker="x", s=20)
        ax_s.scatter(time_bankrupt, s_bankrupt, c=self.colours["bankruptcy"], marker="x", s=20)

        # ax_w - workers
        ax_w.plot(time_values, w.T)
        ax_w.set(xlabel="Time", ylabel="Log Count", title="Workers", yscale="log")
        ax_w.grid()
        
        self._text_save_show(fig, ax_s, "single_companies", xtext=0.05, ytext=0.85)
        
        
    def plot_debt(self):
        """Plot the mean debt and fraction who went bankrupt on twinx and below it debt together with salary, last subplot has debt distribution at final time step. 
        """
        # Preprocess
        self._get_data(self.group_name)
        d = -self.d
        mean_debt = self.d.mean(axis=0)[self.skip_values:]
        median_debt = np.median(d, axis=0)[self.skip_values:]
        mean_salary = self.s.mean(axis=0)[self.skip_values:]
        fraction_bankrupt = (self.went_bankrupt[self.skip_values:] / self.N)
        time_values = np.arange(self.skip_values, self.time_steps)
        d_final = self.d[:, -1]
        
        # Create figure
        fig, (ax, ax1) = plt.subplots(nrows=2, figsize=(10, 8))
        c0 = self.colours["debt"]
        c1 = self.colours["bankruptcy"]
        
        ax.plot(time_values, mean_debt, c=c0, label="Mean debt")
        # ax.plot(time_values, median_debt, c=c0, ls="--", label="Median debt")
        ax.set(xlabel="Time", title="Mean debt and bankruptcies", yscale="linear")
        ax.set_ylabel("Price", color=c0)
        ax.tick_params(axis='y', labelcolor=c0)
        ax.grid()
        
        ax_twin = ax.twinx()
        ax_twin.plot(time_values, fraction_bankrupt, color=c1, label="Fraction bankrupt", alpha=0.6)
        ax_twin.set_ylabel("Fraction bankrupt", color=c1)
        ax_twin.tick_params(axis='y', labelcolor=c1)
        
        # ax1 - Salary and debt
        c2 = self.colours["salary"]
        ax1.plot(time_values, mean_debt, c=c0)
        ax1.set(xlabel="Time", title="Mean salary and debt", yscale="linear")
        ax1.set_ylabel("Mean Debt", color=c0)
        ax1.tick_params(axis='y', labelcolor=c0)
        ax1.grid()
        
        ax1_twin = ax1.twinx()
        ax1_twin.plot(time_values, mean_salary, c=c2, alpha=0.7)
        ax1_twin.set_ylabel("Log mean salary", color=c2)
        ax1_twin.tick_params(axis='y', labelcolor=c2)
        ax1_twin.set_yscale("linear")
        
        # ax2 - Debt distribution
        # Nbins = int(np.sqrt(self.N))
        # ax2.hist(d_final, bins=Nbins, color=c0)
        # ax2.set(title="Debt distribution at final time step", xlabel="Debt", ylabel="Counts", yscale="log")
        # ax2.grid()
        
        # Log scale hist requires only positive values
        # self._xlog_hist(d_final, fig, ax2, xlabel="Log Debt", ylabel="Counts", title="Debt distribution at final time step")
        
        self._text_save_show(fig, ax, "debt", xtext=0.05, ytext=0.85, fontsize=6)
    
    
    def plot_mu_mean_s_diversity(self):
        """Plot the system money spent mu over time, and the mean salary on a twinx axis.
        """
        # Get data
        self._get_data(self.group_name)
        _, diversity = self._worker_diversity()  # Already time skipped
        time_values, mu, s = self._skip_values(self.time_values, self.mu, self.s)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(time_values, mu/self.W - s.mean(axis=0), c=self.colours["mu"], label=r"$\mu / W - \hat{s}$", alpha=0.9)
        ax.tick_params(axis='y', labelcolor=self.colours["mu"])
        ax.set_ylabel(r"$\mu / W - \hat{s}$", color=self.colours["mu"])
        ax.set(xlabel="Time", )
        ax.grid()
        
        # Diversity
        twinx = ax.twinx()
        twinx.plot(time_values, diversity, label="Diversity", c=self.colours["diversity"], alpha=0.7)
        twinx.tick_params(axis='y', labelcolor=self.colours["diversity"])
        twinx.set_ylabel("Diversity", color=self.colours["diversity"])
        twinx.set(ylim=(0, self.N))
        
        # Custom legend containing all lines
        # handles, labels = [], []
        # for line in ax.get_lines():
        #     handles.append(line)
        #     labels.append(line.get_label())
        # for line in twinx.get_lines():
        #     handles.append(line)
        #     labels.append(line.get_label())
        # ax.legend(handles, labels, ncols=len(labels), bbox_to_anchor=(0.5, 0.9), loc="lower center",)
        
        # Text, save show
        self._text_save_show(fig, ax, "mu_mean_diversity", xtext=0.05, ytext=0.85)
            
    
    def plot_mutations(self):
        """Compare mean and median salary to the sum of mutations to see if salary is just the noise from mutations.
        """
        # Preprocess
        # Only plot positive mutation contributions
        self._get_data(self.group_name)
        mutation_pos_idx = self.mutations > 0
        mutation_mean_pos = self.mutations[mutation_pos_idx] / self.N
        x_pos = np.arange(self.time_steps)[mutation_pos_idx]   
        
        s_mean = self.s.mean(axis=0)
        s_median = np.median(self.s, axis=0)
        
        # Create figure
        fig, (ax, ax_med) = plt.subplots(nrows=2)
        c0 = self.colours["salary"]
        c1 = self.colours["mutations"]
        ax.plot(s_mean, label="Salary", alpha=0.9, c=c0)
        ax.plot(x_pos, mutation_mean_pos, ".", label="Mean of mutations", alpha=0.5, c=c1)
        ax.set(title="Mean salary", ylabel="Log Price", yscale="log")
        ax.grid()
        
        # ax_med - Median salary
        ax_med.plot(s_median, label="Median salary", alpha=0.9, c=self.colours["salary"], ls="-")
        ax_med.plot(x_pos, mutation_mean_pos, ".", label="Mean of mutations", alpha=0.5, c=c1)
        ax_med.set(title="Median salary", xlabel="Time", ylabel="Log Price", yscale="log")
        ax_med.grid()

        self._add_legend(ax, ncols=3)
        self._text_save_show(fig, ax, "mutations", xtext=0.05, ytext=0.8)


    def plot_multiple_mutation_size_and_minmax(self, group_name_list):
        """Plot the mean salary on one subplot, and the min and max on another subplot for different mutation sizes
        """       
        # Load data
        self._get_data(self.group_name)
        time_vals = np.arange(self.skip_values, self.time_steps)
        mean_salary_list = np.zeros((len(group_name_list), len(time_vals)))
        mutation_size_list = np.zeros(len(group_name_list))
        for i, gname in enumerate(group_name_list):
            self._get_data(gname)
            mean_salary = np.mean(self.s[:, self.skip_values:], axis=0)
            mean_salary_list[i] = mean_salary
            mutation_size_list[i] = self.m
            
        # Get min and max salary, divide by mutation size
        min_salary = np.min(mean_salary_list, axis=1) / mutation_size_list
        max_salary = np.max(mean_salary_list, axis=1) / mutation_size_list
        
        # Create figure
        fig, (ax, ax1) = plt.subplots(nrows=2)
        c_list = general_functions.list_of_colors
        for i, (mean_salary, m) in enumerate(zip(mean_salary_list, mutation_size_list)):
            ax.plot(time_vals, mean_salary, label=f"m={m}", c=c_list[i], alpha=0.7)
            ax1.plot(mutation_size_list, min_salary, "v", c=c_list[i])
            ax1.plot(mutation_size_list, max_salary, "^", c=c_list[i])
        
        # Setup
        ax.set(xlabel="Time", ylabel="Log Price", title="Mean salary for different mutation sizes", yscale="log")
        ax1.set(xlabel="Mutation size", ylabel="Log Price", title="Min and max of mean salary divided by m", yscale="log", xscale="log")
        ax.grid()
        ax1.grid()
        
        self._add_legend(ax, ncols=len(group_name_list)//2, fontsize=5)
        self._text_save_show(fig, ax, "multiple_mutation_size_minmax", xtext=0.05, ytext=0.75)
    
    
    def plot_min_max_vs_m(self, group_name_arr, data_name: str):
        """Plot the mean of repeated measurements of the minimum and maximum of the mean salary, together with their uncertanties.
        """
        # Calculate mean and std of min and max salary for each m
        self._get_data(group_name_arr[0, 0])
        mean_salary_arr, variable_dict = self._load_multiple_variable_repeated(group_name_arr, data_name)
        m_vals = variable_dict["m"]
        
        N_repeats = np.shape(mean_salary_arr)[1]
        time_steps = np.shape(mean_salary_arr)[2]
        
        min_arr = np.min(mean_salary_arr, axis=2)
        mean_min_arr = np.mean(min_arr, axis=1) / m_vals  # Normalize the minimum salary by the mutation magnitude
        std_mean_min_arr = np.std(min_arr, axis=1, ddof=1) / np.sqrt(N_repeats) / m_vals
        
        max_arr = np.max(mean_salary_arr, axis=2) 
        mean_max_arr = np.mean(max_arr, axis=1) / m_vals  # Normalize the maximum salary by the mutation magnitude
        std_mean_max_arr = np.std(max_arr, axis=1, ddof=1) / np.sqrt(N_repeats) / m_vals

        fig, ax = plt.subplots(figsize=(10, 5))
        label_min = r"$\min(\mu)/(Wm)$"
        label_max = r"$\max(\mu)/(Wm)$"
        ax.errorbar(m_vals, mean_min_arr, yerr=std_mean_min_arr, fmt="v", label=label_min, color="k")
        ax.errorbar(m_vals, mean_max_arr, yerr=std_mean_max_arr, fmt="^", label=label_max, color="k")
        ax.set(xlabel=r"$m$", ylabel="Price", yscale="log", xscale="log")
        # ax.set_title(f"Repeated measurements of min and max salary, N={N_repeats}, t={time_steps}", fontsize=10)
        ax.grid()
        
        self._add_legend(ax, ncols=2, y=0.9, fontsize=15)
        
        # Text, save show
        save_name = "min_max_salary_vs_m"
        # Include the last minimum salary taken from the group name to the save name
        print(self.ds)
        last_s_min = group_name_arr[-1, -1].split("_")[-2]
        combined_save_name = save_name + last_s_min
        self._save_fig(fig, combined_save_name)
        plt.show()


    def plot_multiple_m(self, group_name_list, same_plot=False, time_values_show=500):
        """Plot the mean salary for different m values, each on their own subplot or on the same plot if same_plot is True.
        """       
        # Load data
        self._get_data(group_name_list[0])
        time_vals = np.arange(self.skip_values, self.time_steps)
        mean_salary_list = np.zeros((len(group_name_list), len(time_vals)))
        m_list = np.zeros(len(group_name_list))
        for i, gname in enumerate(group_name_list):
            # Get values
            self._get_data(gname)
            mean_salary = np.mean(self.s[:, self.skip_values:], axis=0)
            # Append values
            mean_salary_list[i] = mean_salary
            m_list[i] = self.m
        
        # Plot all salary means on the same plot
        if same_plot:
            fig, ax = plt.subplots(figsize=(12, 8))
            for i, (mean_salary, m) in enumerate(zip(mean_salary_list, m_list)):
                ax.plot(time_vals[:time_values_show], mean_salary[:time_values_show], label=f"m = {m:.3f}")
            ax.set(title="Mean salary for different m values", xlabel="Time", ylabel="Price", yscale="log")
            ax.grid()
            self._add_legend(ax, ncols=len(group_name_list)//2, y=0.9)
            save_ax = ax
        
        else:
            # Create figure
            # Calculate nrows and ncols
            nrows = 2
            ncols = (len(group_name_list) + nrows - 1) // nrows
            fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 8))
            
            for i, (mean_salary, m) in enumerate(zip(mean_salary_list, m_list)):
                ax = axs[i//ncols, i%ncols]
                ax.plot(time_vals, mean_salary, c=self.colours["salary"])
                ax.set_title(fr"m = {m:.3f}", fontsize=8)
                ax.grid()

                # Axis labels. Only the bottom row should have x labels, and only the left column should have y labels
                subplot_spec = ax.get_subplotspec()
                if subplot_spec.is_last_row():
                    ax.set_xlabel("Time")
                if subplot_spec.is_first_col():
                    ax.set_ylabel("Price")
            
            fig.suptitle("Mean salary for different m values")
            save_ax = axs[0, 0]
        
        # Text, save show,
        self._text_save_show(fig, save_ax, f"multiple_m_same_plot{same_plot}", xtext=0.05, ytext=0.75, fontsize=6)

        
    def plot_multiple_alpha(self, group_name_list, same_plot=False):
        """Plot the mean salary for different probability exponents each on their own subplot
        """       
        # Load data
        self._get_data(self.group_name)
        time_vals = np.arange(self.skip_values, self.time_steps)
        mean_salary_list = np.zeros((len(group_name_list), len(time_vals)))
        prob_expo_list = np.zeros(len(group_name_list))
        for i, gname in enumerate(group_name_list):
            self._get_data(gname)
            mean_salary = np.mean(self.s[:, self.skip_values:], axis=0)
            prob_expo = self.prob_expo
            mean_salary_list[i] = mean_salary
            prob_expo_list[i] = prob_expo
        
        # Create figure
        c_list = general_functions.list_of_colors[:len(group_name_list)]

        if same_plot:
            fig, ax = plt.subplots(figsize=(10, 5))
            for i, (mean_salary, expo) in enumerate(zip(mean_salary_list, prob_expo_list)):
                ax.plot(time_vals, mean_salary, label=f"Exponent = {int(expo)}", c=c_list[i])
            ax.set(xlabel="Time", ylabel="Price", yscale="linear")
            ax.set_title("Mean salary for different probability exponents")
            ax.grid()
            self._add_legend(ax, ncols=len(group_name_list)//2, y=0.9)
            save_ax = ax
            
        else:
            # Calculate nrows and ncols
            nrows = 2
            ncols = (len(group_name_list) + nrows - 1) // nrows
            fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 8))
            
            for i, (mean_salary, expo) in enumerate(zip(mean_salary_list, prob_expo_list)):
                ax = axs[i//ncols, i%ncols]
                ax.plot(time_vals, mean_salary, c=self.colours["salary"])
                ax.set_title(fr"$\alpha =$ {int(expo)}")
                ax.set(yscale="linear")
                ax.grid()

                # Axis labels. Only the bottom row should have x labels, and only the left column should have y labels
                subplot_spec = ax.get_subplotspec()
                if subplot_spec.is_last_row():
                    ax.set_xlabel("Time")
                if subplot_spec.is_first_col():
                    ax.set_ylabel("Price")
            save_ax = axs[0, 0]
        
        # Text, save show,
        self._text_save_show(fig, save_ax, f"multiple_prob_expo_same_plot{same_plot}", xtext=0.05, ytext=0.75, fontsize=6)


    def plot_alpha_frequency(self, group_name_list):
        """Using the Power Spectral Density to find the frequency of the ds values in ds_list and plot them
        """
        # Load data
        self._get_data(group_name_list[0])
        time_vals = np.arange(self.skip_values, self.time_steps)
        mean_salary_list = np.zeros((len(group_name_list), len(time_vals)))
        alpha_list = np.zeros(len(group_name_list))
        for i, gname in enumerate(group_name_list):
            # Get values
            self._get_data(gname)
            mean_salary = np.mean(self.s[:, self.skip_values:], axis=0)
            alpha = self.prob_expo
            # Append values
            mean_salary_list[i] = mean_salary
            alpha_list[i] = alpha
        
        # For each data set, using PSD find the frequency of the oscillation by taking the max frequency of the two most dominant frequencies
        freq_list = np.zeros(len(alpha_list))
        freq2_list = np.zeros(len(alpha_list))
        for i, mean_salary in enumerate(mean_salary_list):
            freq, psd = self._PSD_on_dataset(mean_salary, number_of_frequencies=2, fs=1)
            # freq_list has the most prominent frequency, and freq2_list has the second most prominent frequency
            freq_list[i] = freq[0]
            freq2_list[i] = freq[1]
        
        # Linear fit to dominant frequency data        
        par, cov = np.polyfit(alpha_list, freq_list, deg=1, cov=True)
        std = np.sqrt(np.diag(cov))
        x_fit = np.linspace(np.min(alpha_list), np.max(alpha_list), 100)
        y_fit = par[0] * x_fit + par[1]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(alpha_list, freq_list, "o", label="Most prominent frequency")
        ax.plot(alpha_list, freq2_list, "x", label="Second most prominent frequency")
        ax.plot(x_fit, y_fit, ls="--", label=r"Linear fit")
        ax.set(xlabel=r"$\alpha$", ylabel="Frequency", title=r"Mean salary oscillation frequency")
        self._add_legend(ax, ncols=3, x=0.5, y=0.95)
        ax.grid()
        
        # Print the fit parameters with their uncertainty
        fit_text = fr"$a = $ {par[0]:.3f} $\pm$ {std[0]:.3f}, $b = $ {par[1]:.2f} $\pm$ {std[1]:.5f}"
        ax.text(0.95, 0.85, fit_text, transform=ax.transAxes, fontsize=8, horizontalalignment="right")
        print(fit_text)
        
        # Text, save show,
        self._text_save_show(fig, ax, "alpha_frequency", xtext=0.05, ytext=0.75, fontsize=7)


    def plot_alpha_power_spectrum(self, group_name_list):
        # Load data
        self._get_data(group_name_list[0])
        time_vals = np.arange(self.skip_values, self.time_steps)
        mean_salary_list = np.zeros((len(group_name_list), len(time_vals)))
        alpha_list = np.zeros(len(group_name_list))
        for i, gname in enumerate(group_name_list):
            # Get values
            self._get_data(gname)
            mean_salary = np.mean(self.s[:, self.skip_values:], axis=0)
            alpha = self.prob_expo 
            # Append values
            mean_salary_list[i] = mean_salary
            alpha_list[i] = ds
        
        psd_list = []
        freq_list = []
        for i, mean_salary in enumerate(mean_salary_list):
            freq, psd = self._compute_PSD(mean_salary, fs=1)
            freq_list.append(freq)
            psd_list.append(psd)
        
        # Create figure. One subplot for each ds value and corresponding data set
        nrows = 2
        ncols = (len(group_name_list) + nrows - 1) // nrows
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 8))
        
        for i, (psd, freq, ds) in enumerate(zip(psd_list, freq_list, alpha_list)):
            ax = axs[i//ncols, i%ncols]
            ax.semilogy(freq, psd)
            ax.set_title(f"ds = {ds:.3f}", fontsize=7)
            ax.grid()

            # Axis labels. Only the bottom row should have x labels, and only the left column should have y labels
            subplot_spec = ax.get_subplotspec()
            if subplot_spec.is_last_row():
                ax.set_xlabel("Frequency")
            if subplot_spec.is_first_col():
                ax.set_ylabel("Power Spectral Density")
                
            # Insert a zoomed in plot of the first peak
            axins = inset_axes(ax, "30%", "30%", loc="upper right")
            axins.semilogy(freq, psd)
            # Determine limits
            peak_idx = np.argmax(psd)
            number_of_points_to_show = 20
            x1, x2 = -0.001, 0.05
            mask = np.logical_and(freq >= x1, freq <= x2)
            y1, y2 = np.min(psd[mask]), np.max(psd[mask])
            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)
            axins.set_xticks([])
            axins.set_yticks([])
            mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
            
        
        fig.suptitle(r"Power Spectral Density of mean salary for different $\alpha$ values")
        # Text, save show,
        self._text_save_show(fig, axs[0, 0], "alpha_power_spectrum", xtext=0.05, ytext=0.75, fontsize=6)
        

    def plot_multiple_s_min(self, group_name_list):
        """Plot the mean salary for minimum salary values
        """       
        # Load data
        self._get_data(group_name_list[0])
        time_vals = np.arange(self.skip_values, self.time_steps)
        mean_salary_list = np.zeros((len(group_name_list), len(time_vals)))
        bankruptcy_list = np.zeros((len(group_name_list), len(time_vals)))
        s_min_list = np.zeros(len(group_name_list))
        for i, gname in enumerate(group_name_list):
            # Get values
            self._get_data(gname)
            mean_salary = np.mean(self.s[:, self.skip_values:], axis=0)
            # Append values
            mean_salary_list[i] = mean_salary
            print(self.s_min)
            s_min_list[i] = self.s_min * 1
            bankruptcy_list[i] = self.went_bankrupt[self.skip_values:] / self.N

        # Create figure
        # Calculate nrows and ncols
        nrows = 2
        ncols = (len(group_name_list) + nrows - 1) // nrows
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 12))
        
        for i, (mean_salary, s_min, p_bank) in enumerate(zip(mean_salary_list, s_min_list, bankruptcy_list)):
            ax = axs[i//ncols, i%ncols]
            
            twin_x = ax.twinx()
            # twin_x.plot(time_vals, p_bank, c=self.colours["bankruptcy"], label="Fraction bankrupt", alpha=0.25)
            twin_x.tick_params(axis='y', labelcolor=self.colours["bankruptcy"])
            
            ax.plot(time_vals, mean_salary, c=self.colours["salary"])
            # ax.plot(self.time_values, median_salary, c="k", ls="dotted", alpha=0.75)
            ax.tick_params(axis='y', labelcolor=self.colours["salary"])
            
            title = r"$s_\text{min} = $" + f"{s_min:.0e}"
            ax.set_title(title, fontsize=8)
            ax.set(yscale="log")
            ax.grid()

            # Axis labels. Only the bottom row should have x labels, and only the left column should have y labels
            subplot_spec = ax.get_subplotspec()
            if subplot_spec.is_last_row():
                ax.set_xlabel("Time")
            if subplot_spec.is_first_col():
                ax.set_ylabel("Log Price")
        
        fig.suptitle(fr"Salary for different minimum salary values, $m={self.m}$")
        
        # save show
        self._save_fig(fig, "multiple_s_min")
        plt.show()
    
    
    def _load_multiple_variable_repeated(self, group_name_arr, data_name: str) -> tuple:
        """Loop over the 2d array group_name_arr and store the mean salary for each group name in an array.

        Args:
            group_name_arr (np.ndarray): 2d array with group names. Rows are variable values, columns are repeated runs.
            
        Returns:
            data_arr (np.ndarray): Array with the data values for each group name.
            variable_dict (dict): Dictionary with the variable values for each group name.
        """
        assert data_name in ["salary", "mu"]
        
        time_vals = np.arange(self.skip_values, self.time_steps)
        data_arr = np.zeros((group_name_arr.shape[0], group_name_arr.shape[1], len(time_vals)))
        ds_arr = np.zeros(group_name_arr.shape[0])
        m_arr = np.zeros(group_name_arr.shape[0])
        alpha_arr = np.zeros(group_name_arr.shape[0])
        
        for i in range(group_name_arr.shape[0]):
            for j in range(group_name_arr.shape[1]):
                gname = group_name_arr[i, j]
                self._get_data(gname)
                if data_name == "salary":
                    data = np.mean(self.s[:, self.skip_values:], axis=0)
                elif data_name == "mu":
                    data = self.mu[self.skip_values:]
                data_arr[i, j] = data
            ds_arr[i] = self.ds
            m_arr[i] = self.m
            alpha_arr[i] = self.prob_expo
        
        variable_dict = {"ds": ds_arr, "m": m_arr, "alpha": alpha_arr}
        return data_arr, variable_dict
    

    def plot_multiple_ds(self, group_name_list, same_plot=False):
        """Plot the mean salary for different ds values, each on their own subplot
        """       
        # Load data
        self._get_data(group_name_list[0])
        time_vals = np.arange(self.skip_values, self.time_steps)
        mean_salary_list = np.zeros((len(group_name_list), len(time_vals)))
        ds_list = np.zeros(len(group_name_list))
        for i, gname in enumerate(group_name_list):
            # Get values
            self._get_data(gname)
            mean_salary = np.mean(self.s[:, self.skip_values:], axis=0)
            ds = self.ds
            # Append values
            mean_salary_list[i] = mean_salary
            ds_list[i] = ds
        
        # Plot all salary means on the same plot
        if same_plot:
            fig, ax = plt.subplots(figsize=(12, 8))
            for i, (mean_salary, ds) in enumerate(zip(mean_salary_list, ds_list)):
                ax.plot(time_vals[:500], mean_salary[:500], label=f"ds = {ds:.3f}")
            ax.set(title="Mean salary for different ds values", xlabel="Time", ylabel="Price")
            ax.grid()
            self._add_legend(ax, ncols=len(group_name_list)//2, y=0.9)
            save_ax = ax
        
        else:
            # Create figure
            # Calculate nrows and ncols
            nrows = 2
            ncols = (len(group_name_list) + nrows - 1) // nrows
            fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 8))
            
            for i, (mean_salary, ds) in enumerate(zip(mean_salary_list, ds_list)):
                ax = axs[i//ncols, i%ncols]
                ax.plot(time_vals, mean_salary, c=self.colours["salary"])
                ax.set_title(fr"ds = {ds:.3f}", fontsize=8)
                ax.set(yscale="linear")
                ax.grid()

                # Axis labels. Only the bottom row should have x labels, and only the left column should have y labels
                subplot_spec = ax.get_subplotspec()
                if subplot_spec.is_last_row():
                    ax.set_xlabel("Time")
                if subplot_spec.is_first_col():
                    ax.set_ylabel("Price")
            
            fig.suptitle("Mean salary for different ds values")
            save_ax = axs[0, 0]
        
        # Text, save show,
        self._text_save_show(fig, save_ax, f"multiple_ds_same_plot{same_plot}", xtext=0.05, ytext=0.75, fontsize=6)


    def plot_ds_frequency(self, group_name_list):
        """Using the Power Spectral Density to find the frequency of the ds values in ds_list and plot them
        """
        # Load data
        self._get_data(group_name_list[0])
        time_vals = np.arange(self.skip_values, self.time_steps)
        mean_salary_list = np.zeros((len(group_name_list), len(time_vals)))
        ds_list = np.zeros(len(group_name_list))
        for i, gname in enumerate(group_name_list):
            # Get values
            self._get_data(gname)
            mean_salary = np.mean(self.s[:, self.skip_values:], axis=0)
            ds = self.ds
            # Append values
            mean_salary_list[i] = mean_salary
            ds_list[i] = ds
        
        # For each data set, using PSD find the frequency of the oscillation by taking the max frequency of the two most dominant frequencies
        freq_list = np.zeros(len(ds_list))
        freq2_list = np.zeros(len(ds_list))
        for i, mean_salary in enumerate(mean_salary_list):
            freq, psd = self._PSD_on_dataset(mean_salary, number_of_frequencies=2, fs=1)
            # freq_list has the most prominent frequency, and freq2_list has the second most prominent frequency
            freq_list[i] = freq[0]
            freq2_list[i] = freq[1]
        
        # Linear fit to dominant frequency data        
        par, cov = np.polyfit(ds_list[:-3], freq_list[:-3], deg=1, cov=True)
        std = np.sqrt(np.diag(cov))
        x_fit = np.linspace(np.min(ds_list), np.max(ds_list), 100)
        y_fit = par[0] * x_fit + par[1]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(ds_list, freq_list, "o", label="Most prominent frequency")
        ax.plot(ds_list, freq2_list, "x", label="Second most prominent frequency")
        ax.plot(x_fit, y_fit, ls="--", label=r"Linear fit (excluding high $\Delta s$)")
        ax.set(xlabel=r"$\Delta s$", ylabel="Frequency", title=r"Mean salary oscillation frequency")
        self._add_legend(ax, ncols=3, x=0.5, y=0.95)
        ax.grid()
        
        # Print the fit parameters with their uncertainty
        fit_text = fr"$a = $ {par[0]:.3f} $\pm$ {std[0]:.3f}, $b = $ {par[1]:.2f} $\pm$ {std[1]:.5f}"
        ax.text(0.95, 0.85, fit_text, transform=ax.transAxes, fontsize=8, horizontalalignment="right")
        print(fit_text)
        
        # Text, save show,
        self._text_save_show(fig, ax, "ds_frequency", xtext=0.05, ytext=0.75, fontsize=7)


    def get_PSD_freq_multiple_var(self, group_name_arr: list, var_name: str, data_name: str):
        """Repeatable measurements of the frequency using PSD for a variable.

        Args:
            group_name_arr (list): _description_
            var_name (str): _description_
            data_name (str): _description_

        Returns:
            tuple: mean_freq1, std_freq1, mean_freq2, std_freq2, var_vals
        """
        # Load data        
        self._get_data(group_name_arr[0, 0])
        data_arr, variable_dict = self._load_multiple_variable_repeated(group_name_arr, data_name)
        var_vals = variable_dict[var_name]
        N_variable_values = data_arr.shape[0]
        N_repeats = data_arr.shape[1]
        
        # For each variable value, loop over repeats and store the PSD frequency
        freq1_arr = np.zeros((N_variable_values, N_repeats))  # Dominant PSD frequency
        freq2_arr = np.zeros((N_variable_values, N_repeats))  # Second dominant PSD frequency

        for i in range(N_variable_values):
            for j in range(N_repeats):
                data_set = data_arr[i, j]
                # If the dataset contains Inf or NaNs, store NaN frequencies
                if np.any(np.isinf(data_set)) or np.any(np.isnan(data_set)):
                    freq1_arr[i, j] = np.nan
                    freq2_arr[i, j] = np.nan
                    continue
                
                # Calculate and store PSD frequency
                freq, psd = self._PSD_on_dataset(data_set, number_of_frequencies=2, fs=1)
                freq1_arr[i, j] = freq[0]
                freq2_arr[i, j] = freq[1]

        # Calculate mean and std of the PSD frequencies, ignoring NaNs
        mean_freq1 = np.nanmean(freq1_arr, axis=1)
        std_freq1 = np.nanstd(freq1_arr, axis=1, ddof=1) / np.sqrt(N_repeats)
        mean_freq2 = np.nanmean(freq2_arr, axis=1)
        std_freq2 = np.nanstd(freq2_arr, axis=1, ddof=1) / np.sqrt(N_repeats)
        
        # Check if any of the errors have almost-0 values, if so, give them the std of the mean as error
        std_freq1 = np.where(std_freq1<1e-8, np.std(mean_freq1), std_freq1)
        
        return mean_freq1, std_freq1, mean_freq2, std_freq2, var_vals


    def linear_fit(self, x, y, std_y, print_results=False):
        # Remove NaNs
        is_nan_idx = np.isnan(y)
        x = x[~is_nan_idx]
        y = y[~is_nan_idx]
        std_y = std_y[~is_nan_idx]
        
        # Define linear function
        linear = lambda x, a, b: a * x + b
        # Perform fit
        p0 = [0.15, 0.05]
        par, cov = scipy.optimize.curve_fit(linear, x, y, p0=p0, sigma=std_y)
        par_std = np.sqrt(np.diag(cov))
        # Get p-value
        chi2_val = general_functions.chi2(y, linear(x, *par), std_y)
        Ndof = len(x) - len(par)
        p_value = scipy.stats.chi2.sf(chi2_val, Ndof)
        
        x_fit = np.linspace(np.min(x), np.max(x), 200)
        y_fit = par[0] * x_fit + par[1]
        
        if print_results:
            # Print the fit parameters with their uncertainty, and the p-value
            fit_text = fr"$a = $ {par[0]:.3f} $\pm$ {par_std[0]:.3f}, $b = $ {par[1]:.2f} $\pm$ {par_std[1]:.5f}, $p=${p_value:.8f}"
            print(fit_text)
            print(f"P(chi2={chi2_val:.2f}, Ndof={Ndof}) = {p_value:.4f}")
        
        prop_dict = {"a": par[0], "b": par[1], "std_a": par_std[0], "std_b": par_std[1], "p_value": p_value, "Ndof": Ndof, "chi2": chi2_val}
        
        return x_fit, y_fit, prop_dict


    def plot_ds_frequency_multiple_datasets(self, group_name_tensor, data_name: str):
        """Plot multiple datasets (such as alpha=1, alpha=4) on the same plot, each with their own frequency and linear fit.

        Args:
            group_name_tensor (_type_): _description_
            data_name (str): _description_
        """
        # For each dataset in group_name_tensor, find the frequency of the oscillation in the data set and perform a linear fit to it.
        # Plot all the frequencies and fits on the same plot.
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 5))
        fit_results = np.zeros((len(group_name_tensor), 2), dtype=object)
        marker_list = ["x", "+", "*", ".", "h", "d"]  # general_functions.list_of_markers
        color_list = general_functions.list_of_colors
        number_of_markers = len(marker_list)
        shift_factor = 0.00
        var_shift_factor_list = np.linspace(-shift_factor, shift_factor, len(group_name_tensor))   # To prevent all points being on top of each other, shift their x-values a tiny bit
        
        for i, group_name_arr in enumerate(group_name_tensor):
            mean_freq1, std_freq1, mean_freq2, std_freq2, var_list = self.get_PSD_freq_multiple_var(group_name_arr, "ds", data_name)
            x_fit, y_fit, prop_dict = self.linear_fit(var_list, mean_freq1, std_freq1)
            a, std_a, b, std_b, p = prop_dict["a"], prop_dict["std_a"], prop_dict["b"], prop_dict["std_b"], prop_dict["p_value"]
            
            # Get the alpha, N and W values from the group name
            par_dict = self._get_par_from_name(group_name_arr[0, 0])
            alpha, N, W = par_dict["alpha"], par_dict["N"], par_dict["W"]

            # To prevent all points being on top of each other, shift their x-values a tiny bit. Go from 98% of the width to 102% of the width
            var_list_plot = var_list + var_shift_factor_list[i] * (var_list[-1] - var_list[0])
            
            # Plot data
            ax.errorbar(var_list_plot, mean_freq1, yerr=std_freq1, c=color_list[i % number_of_markers], fmt=marker_list[-(i % number_of_markers)], 
                        label=fr"$\alpha=${alpha}, $N=${N}, $W=${W}", markersize=8, alpha=0.9)
            # ax.errorbar(var_list, mean_freq2, yerr=std_freq2, c=color_list[i], fmt=marker_list[-i-1], label=fr"Second frequency, $\alpha=${alpha}, $N=${N}, $W=${W}")
            
            # Plot fit
            fit_label = fr"Fit, $a = {a:.3f} \pm {std_a:.3f}, b = {b:.2f} \pm {std_b:.5f}, p = {p:.4f}$"
            ax.plot(x_fit, y_fit, ls="--", c=color_list[i % number_of_markers], )#label=fit_label, )
            fit_results[i, 0] = fit_label
            fit_results[i, 1] = par_dict
        
        # Axis setup
        ax.set(xlabel=r"$\Delta s$", ylabel="Frequency")
        ax.grid()
        ax.legend(ncols=2, bbox_to_anchor=(0.025, 0.975), loc="upper left", fontsize=12)
        # self._add_legend(ax, ncols=2, x=0.4, y=0.85, fontsize=12)
        
        # Text save show
        self._text_save_show(fig, ax, f"ds_frequency_multiple_dataset", xtext=0.05, ytext=0.75, fontsize=8)
        
        # Print the fit results
        for i in range(len(fit_results)):
            fit_label = fit_results[i, 0]
            par_dict = fit_results[i, 1]
            print(f"System parameters: alpha = {par_dict['alpha']}, N = {par_dict['N']}, W = {par_dict['W']}")
            print(fit_label)
            print("")
            


    def plot_var_frequency(self, group_name_arr: list, var_name: str, data_name: str, points_to_exclude_from_fit=0, show_second_dominant_freq=False, show_fit_results=False):
        """Use the PSD to find the frequency of the oscillation in the data set for different var_name values.

        Args:
            group_name_list (list): _description_
            var_name (str): Either "ds" or "alpha". Determines what variable to plot against.
            data_name (str): Either "salary" or "mu". Determines what data to load and find the frequency on.
        """
        assert var_name in ["ds", "alpha"], f"var_name must be either 'ds' or 'alpha', not {var_name}"
        assert data_name in ["salary", "mu"], f"data_name must be either 'salary' or 'mu', not {data_name}"

        # Load data        
        self._get_data(group_name_arr[0, 0])
        data_arr, variable_dict = self._load_multiple_variable_repeated(group_name_arr, data_name)
        var_list = variable_dict[var_name]
        N_variable_values = data_arr.shape[0]
        N_repeats = data_arr.shape[1]
        time_steps = data_arr.shape[2]
        
        # For each variable value, loop over repeats and store the PSD frequency
        freq1_arr = np.zeros((N_variable_values, N_repeats))  # Dominant PSD frequency
        freq2_arr = np.zeros((N_variable_values, N_repeats))  # Second dominant PSD frequency

        for i in range(N_variable_values):
            for j in range(N_repeats):
                freq, psd = self._PSD_on_dataset(data_arr[i, j], number_of_frequencies=2, fs=1)
                freq1_arr[i, j] = freq[0]
                freq2_arr[i, j] = freq[1]

        # Calculate mean and std of the PSD frequencies
        mean_freq1 = np.mean(freq1_arr, axis=1)
        std_freq1 = np.std(freq1_arr, axis=1, ddof=1) / np.sqrt(N_repeats)
        mean_freq2 = np.mean(freq2_arr, axis=1)
        std_freq2 = np.std(freq2_arr, axis=1, ddof=1) / np.sqrt(N_repeats)
        
        # Check if any of the errors have almost-0 values, if so, give them the std of the mean as error
        std_freq1 = np.where(std_freq1<1e-8, np.std(mean_freq1), std_freq1)
        
        # Linear function defined using lambda
        linear = lambda x, a, b: a * x + b
        p0 = [0.15, 0.05]
        par, cov = scipy.optimize.curve_fit(linear, var_list, mean_freq1, p0=p0, sigma=std_freq1)
        par_std = np.sqrt(np.diag(cov))
        # Get p-value
        chi2_val = general_functions.chi2(mean_freq1, linear(var_list, *par), std_freq1)
        Ndof = len(var_list) - len(par)
        p_value = scipy.stats.chi2.sf(chi2_val, Ndof)
        
        x_fit = np.linspace(np.min(var_list), np.max(var_list), 200)
        y_fit = par[0] * x_fit + par[1]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.errorbar(var_list, mean_freq1, yerr=std_freq1, fmt="o", label="Most prominent frequency")
        if show_second_dominant_freq: ax.errorbar(var_list, mean_freq2, yerr=std_freq2, fmt="x", label="Second most prominent frequency")
        ax.plot(x_fit, y_fit, ls="--", c="k", label=r"Linear fit")
        if var_name == "ds":
            ax.set(xlabel=r"$\Delta s$", ylabel="Frequency", )
        elif var_name == "alpha":
            ax.set(xlabel=r"$\alpha$", ylabel="Frequency", )
        title=fr"{data_name} oscillation frequency"
        # ax.set_title(title)
        print(title)
        
        self._add_legend(ax, ncols=3, x=0.5, y=0.9, fontsize=14)
        ax.grid()
        
        # Print the fit parameters with their uncertainty
        fit_text = fr"$a = $ {par[0]:.3f} $\pm$ {par_std[0]:.3f}, $b = $ {par[1]:.2f} $\pm$ {par_std[1]:.5f}, $p=${p_value:.8f}"
        if show_fit_results: ax.text(0.95, 0.85, fit_text, transform=ax.transAxes, fontsize=8, horizontalalignment="right")
        print(fit_text)
        print(f"P(chi2={chi2_val:.2f}, Ndof={Ndof}) = {p_value:.4f}")
        
        # Text, save show,
        self._text_save_show(fig, ax, f"frequency_{var_name}_{data_name}", xtext=0.05, ytext=0.75, fontsize=8)


    def plot_power_spectrum(self, group_name_list):
        # Load data
        self._get_data(group_name_list[0])
        time_vals = np.arange(self.skip_values, self.time_steps)
        mean_salary_list = np.zeros((len(group_name_list), len(time_vals)))
        ds_list = np.zeros(len(group_name_list))
        N_list = np.zeros(len(group_name_list))
        W_list = np.zeros(len(group_name_list))
        alpha_list = np.zeros(len(group_name_list))
        for i, gname in enumerate(group_name_list):
            # Get values
            self._get_data(gname)
            mean_salary = self.mu[self.skip_values:] / self.W
            ds = self.ds
            # Append values
            mean_salary_list[i] = mean_salary
            ds_list[i] = ds
            # Get the alpha, N and W values from the group name
            par_dict = self._get_par_from_name(gname)
            alpha_list[i] = par_dict["alpha"]
            N_list[i] = par_dict["N"]
            W_list[i] = par_dict["W"]
        
        psd_list = []
        freq_list = []
        peak_psd_list = []
        peak_freq_list = []
        for i, mean_salary in enumerate(mean_salary_list):
            freq, psd = self._compute_PSD(mean_salary, fs=1)
            freq_peak, psd_peak = self._find_dominant_frequencies(freq, psd, number_of_frequencies=1)
            freq_list.append(freq)
            psd_list.append(psd)
            peak_psd_list.append(psd_peak)
            peak_freq_list.append(freq_peak)
        
        # Create figure. One subplot for each ds value and corresponding data set
        nrows = 2
        ncols = (len(group_name_list) + nrows - 1) // nrows
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 8))
        
        psd_min, psd_max = np.min(psd_list), np.max(psd_list)
        exponent_range = (np.log10(psd_min), np.log10(psd_max))  # For the y ticklabels
        ylim = (psd_min*0.7, psd_max*1.3)
        
        for i, psd, freq, ds in zip(np.arange(len(psd_list)), psd_list, freq_list, ds_list):
            ax = axs[i//ncols, i%ncols]
            ax.semilogy(freq, psd)
            # title = fr"$\Delta s / s = {ds:.3f}$, $\alpha = {alpha_list[i]:.0f}$, $N = {N_list[i]:.0f}$, $W = {W_list[i]:.0f}$"
            # ax.set_title(title, fontsize=7)
            ax.set(ylim=ylim)
            ax.grid()

            # Axis ticks and ticklabels
            self._axis_log_ticks_and_labels(ax, exponent_range=exponent_range, which="y", labels_skipped=2)
            # Axis labels. Only the bottom row should have x labels, and only the left column should have y labels
            self._axis_labels_outer(ax, x_label="Frequency", y_label="Power Spectral Density")
                
            # Insert a zoomed in plot of the first peak
            axins = inset_axes(ax, "30%", "30%", loc="upper right")
            axins.plot(freq, psd, ".-")
            axins.plot(peak_freq_list[i], peak_psd_list[i], "x", markersize=10, alpha=0.75)
            axins.set(xscale="linear", yscale="log")
            # Determine limits
            peak_idx = np.argmax(psd)
            number_of_points_to_show = 20
            x1, x2 = -0.001, 0.01
            mask = np.logical_and(freq >= x1, freq <= x2)
            y1, y2 = np.min(psd[mask]), np.max(psd[mask])
            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2*1.8)
            axins.set_xticks([])
            axins.set_yticks([])
            mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
        
        # fig.suptitle("Power Spectral Density of mean salary for different ds values")
        # Text, save show,
        self._text_save_show(fig, axs[0, 0], f"power_spectrum_ds{self.ds}", xtext=0.05, ytext=0.75, fontsize=0)


    def plot_min_max_vs_alpha(self, group_name_arr, data_name: str):
        """Plot the mean of repeated measurements of the minimum and maximum of the mean salary, together with their uncertainties.
        """
        # Calculate mean and std of min and max salary for each alpha
        self._get_data(group_name_arr[0, 0])
        mean_salary_arr, variable_dict = self._load_multiple_variable_repeated(group_name_arr, data_name)
        alpha_vals = variable_dict["alpha"]
        
        N_repeats = np.shape(mean_salary_arr)[1]
        time_steps = np.shape(mean_salary_arr)[2]
        
        min_arr = np.min(mean_salary_arr, axis=2)
        mean_min_arr = np.mean(min_arr, axis=1) #/ alpha_vals  # Normalize the minimum salary by the probability exponent
        std_mean_min_arr = np.std(min_arr, axis=1, ddof=1) / np.sqrt(N_repeats) #/ alpha_vals
        
        max_arr = np.max(mean_salary_arr, axis=2)
        mean_max_arr = np.mean(max_arr, axis=1) #/ alpha_vals  # Normalize the maximum salary by the probability exponent
        std_mean_max_arr = np.std(max_arr, axis=1, ddof=1) / np.sqrt(N_repeats) #/ alpha_vals

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.errorbar(alpha_vals, mean_min_arr, yerr=std_mean_min_arr, fmt="v", label=r"$\min(\bar{s})$", color="k")
        ax.errorbar(alpha_vals, mean_max_arr, yerr=std_mean_max_arr, fmt="^", label=r"$\max(\bar{s})$", color="k")
        ax.set(xlabel=r"$\alpha$", ylabel="Price", yscale="log", xscale="linear")
        ax.set_title(f"Repeated measurements of min and max {data_name}, N={N_repeats}, t={time_steps}", fontsize=10)
        ax.grid()
        
        self._add_legend(ax, ncols=2, y=0.9)
        
        # Text, save show
        save_name = "min_max_salary_vs_alpha"
        # Include the last minimum salary taken from the group name to the save name
        last_s_min = group_name_arr[-1, -1].split("_")[-2]
        combined_save_name = save_name + last_s_min
        self._save_fig(fig, combined_save_name)
        plt.show()
        

    def plot_m_frequency(self, group_name_list):
        """Using the Power Spectral Density to find the frequency of the ds values in ds_list and plot them
        """
        # Load data
        self._get_data(group_name_list[0])
        time_vals = np.arange(self.skip_values, self.time_steps)
        mean_salary_list = np.zeros((len(group_name_list), len(time_vals)))
        m_list = np.zeros(len(group_name_list))
        for i, gname in enumerate(group_name_list):
            # Get values
            self._get_data(gname)
            mean_salary = np.mean(self.s[:, self.skip_values:], axis=0)
            # Append values
            mean_salary_list[i] = mean_salary
            m_list[i] = self.m
        
        # For each data set, using PSD find the frequency of the oscillation by taking the max frequency of the two most dominant frequencies
        freq_list = np.zeros(len(m_list))
        freq2_list = np.zeros(len(m_list))
        for i, mean_salary in enumerate(mean_salary_list):
            freq, psd = self._PSD_on_dataset(mean_salary, number_of_frequencies=2, fs=1)
            # freq_list has the most prominent frequency, and freq2_list has the second most prominent frequency
            freq_list[i] = freq[0]
            freq2_list[i] = freq[1]
        
        # Linear fit to dominant frequency data        
        par, cov = np.polyfit(m_list, freq_list, deg=1, cov=True)
        std = np.sqrt(np.diag(cov))
        x_fit = np.linspace(np.min(m_list), np.max(m_list), 100)
        y_fit = par[0] * x_fit + par[1]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(m_list, freq_list, "o", label="Most prominent frequency")
        ax.plot(m_list, freq2_list, "x", label="Second most prominent frequency")
        ax.plot(x_fit, y_fit, ls="--", label=r"Linear fit (excluding high $\Delta s$)")
        ax.set(xlabel=r"Mutation magnitude $m$", ylabel="Frequency", title=r"Mean salary oscillation frequency")
        self._add_legend(ax, ncols=3, x=0.5, y=0.95)
        ax.grid()
        
        # Print the fit parameters with their uncertainty
        fit_text = fr"$a = $ {par[0]:.3f} $\pm$ {std[0]:.3f}, $b = $ {par[1]:.2f} $\pm$ {std[1]:.5f}"
        ax.text(0.95, 0.85, fit_text, transform=ax.transAxes, fontsize=8, horizontalalignment="right")
        print(fit_text)
        
        # Text, save show,
        self._text_save_show(fig, ax, "m_frequency", xtext=0.05, ytext=0.75, fontsize=7)
        

    def plot_m_power_spectrum(self, group_name_list):
        # Load data
        self._get_data(group_name_list[0])
        time_vals = np.arange(self.skip_values, self.time_steps)
        mean_salary_list = np.zeros((len(group_name_list), len(time_vals)))
        m_list = np.zeros(len(group_name_list))
        for i, gname in enumerate(group_name_list):
            # Get values
            self._get_data(gname)
            mean_salary = np.mean(self.s[:, self.skip_values:], axis=0)
            # Append values
            mean_salary_list[i] = mean_salary
            m_list[i] = self.m
        
        psd_list = []
        freq_list = []
        for i, mean_salary in enumerate(mean_salary_list):
            freq, psd = self._compute_PSD(mean_salary, fs=1)
            freq_list.append(freq)
            psd_list.append(psd)
        
        # Create figure. One subplot for each ds value and corresponding data set
        nrows = 2
        ncols = (len(group_name_list) + nrows - 1) // nrows
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 8))
        
        for i, (psd, freq, ds) in enumerate(zip(psd_list, freq_list, m_list)):
            ax = axs[i//ncols, i%ncols]
            ax.semilogy(freq, psd)
            ax.set_title(fr"$m =$ {ds:.3f}", fontsize=7)
            ax.grid()

            # Axis labels. Only the bottom row should have x labels, and only the left column should have y labels
            subplot_spec = ax.get_subplotspec()
            if subplot_spec.is_last_row():
                ax.set_xlabel("Frequency")
            if subplot_spec.is_first_col():
                ax.set_ylabel("Power Spectral Density")
                
            # Insert a zoomed in plot of the first peak
            axins = inset_axes(ax, "30%", "30%", loc="upper right")
            axins.semilogy(freq, psd)
            # Determine limits
            x1, x2 = -0.001, 0.05
            mask = np.logical_and(freq >= x1, freq <= x2)
            y1, y2 = np.min(psd[mask]), np.max(psd[mask])
            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)
            axins.set_xticks([])
            axins.set_yticks([])
            mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
            
        
        fig.suptitle("Power Spectral Density of mean salary for different m values")
        # Text, save show,
        self._text_save_show(fig, axs[0, 0], "m_power_spectrum", xtext=0.05, ytext=0.75, fontsize=6)
        

    def plot_N_W_ratio(self, group_name_list, show_bankruptcy):
        """Plot the mean salary for different ds values, each on their own subplot
        """       
        # Load data
        self._get_data(group_name_list[0])
        time_vals = np.arange(self.skip_values, self.time_steps)
        mean_salary_list = np.zeros((len(group_name_list), len(time_vals)))
        bankruptcy_list = np.zeros((len(group_name_list), len(time_vals)))
        N_list = np.zeros(len(group_name_list))
        W_list = np.zeros(len(group_name_list))
        for i, gname in enumerate(group_name_list):
            # Get values
            self._get_data(gname)
            mean_salary = np.mean(self.s[:, self.skip_values:], axis=0)
            bankruptcy = self.went_bankrupt[self.skip_values:] / self.N
            N = self.N
            W = self.W  
            # Append values
            mean_salary_list[i] = mean_salary
            bankruptcy_list[i] = bankruptcy
            N_list[i] = N
            W_list[i] = W
        N_W_ratio = W_list[0] / N_list[0]
        # Create figure
        # Calculate nrows and ncols
        # If there is less than 4 data sets, make 1 row
        less_than_four_datasets = len(group_name_list) < 4
        if less_than_four_datasets:
            nrows = 1
            ncols = len(group_name_list)
        else:
            nrows = 2
            ncols = (len(group_name_list) + nrows - 1) // nrows

        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 8))
        
        for i, (mean_salary, bankruptcy, N, W) in enumerate(zip(mean_salary_list, bankruptcy_list, N_list, W_list)):
            if len(group_name_list) == 1:
                ax = axs
            elif less_than_four_datasets:
                ax = axs[i]
            else:
                ax = axs[i//ncols, i%ncols]
                
            if show_bankruptcy:
                twin_x = ax.twinx()
                twin_x.plot(time_vals, bankruptcy, c=self.colours["bankruptcy"], alpha=0.5)
                twin_x.set_ylabel("Fraction bankrupt", color=self.colours["bankruptcy"])
                twin_x.tick_params(axis='y', labelcolor=self.colours["bankruptcy"])
                
            ax.plot(time_vals, mean_salary, c=self.colours["salary"])
            ax.set_title(fr"$W/N = {W:.0f}/{N:.0f}$", fontsize=8)
            ax.set(yscale="linear", xlim=self.xlim)
            ax.grid()

            # Axis labels. Only the bottom row should have x labels, and only the left column should have y labels
            subplot_spec = ax.get_subplotspec()
            if subplot_spec.is_last_row():
                ax.set_xlabel("Time")
            if subplot_spec.is_first_col():
                ax.set_ylabel("Price")
        
        # Text, save show,
        if len(group_name_list) == 1:
            save_ax = axs
        elif less_than_four_datasets:
            save_ax = axs[0]
        else:
            save_ax = axs[0, 0]
        self._text_save_show(fig, save_ax, "N_W_ratio", xtext=0.05, ytext=0.75, fontsize=6)
        
        
    def plot_N_var_W_const(self, group_name_list):
        """Plot the mean salary for N values, each on their own subplot
        """       
        # Load data
        self._get_data(group_name_list[0])
        time_vals = np.arange(self.skip_values, self.time_steps)
        mean_salary_list = np.zeros((len(group_name_list), len(time_vals)))
        N_list = np.zeros(len(group_name_list))
        W_list = np.zeros_like(N_list)
        for i, gname in enumerate(group_name_list):
            # Get values
            self._get_data(gname)
            mean_salary = np.mean(self.s[:, self.skip_values:], axis=0)
            # Append values
            mean_salary_list[i] = mean_salary
            N_list[i] = self.N
            W_list[i] = self.W
        # Create figure
        # Calculate nrows and ncols
        # If there is less than 4 data sets, make 1 row
        less_than_four_datasets = len(group_name_list) < 4
        if less_than_four_datasets:
            nrows = 1
            ncols = len(group_name_list)
        else:
            nrows = 2
            ncols = (len(group_name_list) + nrows - 1) // nrows
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 8))
        
        for i, (mean_salary, N, W) in enumerate(zip(mean_salary_list, N_list, W_list)):
            # Get current axis
            if len(group_name_list) == 1:
                ax = axs
            elif less_than_four_datasets:
                ax = axs[i]
            else:
                ax = axs[i//ncols, i%ncols]

            ax.plot(time_vals, mean_salary, c=self.colours["salary"])
            ax.set_title(fr"$W/N = {W:.0f}/{N:.0f}$", fontsize=8)
            ax.set(yscale="linear", xlim=self.xlim)
            ax.grid()

            # Axis labels. Only the bottom row should have x labels, and only the left column should have y labels
            subplot_spec = ax.get_subplotspec()
            if subplot_spec.is_last_row():
                ax.set_xlabel("Time")
            if subplot_spec.is_first_col():
                ax.set_ylabel("Price")
        
        # fig.suptitle(fr"Mean salary for $N$ variable, $W = {self.W}$")
        
        # Text, save show,
        if len(group_name_list) == 1:
            save_ax = axs
        elif less_than_four_datasets:
            save_ax = axs[0]
        else:
            save_ax = axs[0, 0]
        self._text_save_show(fig, save_ax, "N_var_W_const", xtext=0.05, ytext=0.75, fontsize=6)


    def plot_N_const_W_var(self, group_name_list):
        """Plot the mean salary for W values, each on their own subplot
        """       
        # Load data
        self._get_data(group_name_list[0])
        time_vals = np.arange(self.skip_values, self.time_steps)
        mean_salary_list = np.zeros((len(group_name_list), len(time_vals)))
        W_list = np.zeros(len(group_name_list))
        for i, gname in enumerate(group_name_list):
            # Get values
            self._get_data(gname)
            mean_salary = np.mean(self.s[:, self.skip_values:], axis=0)
            W = self.W
            # Append values
            mean_salary_list[i] = mean_salary
            W_list[i] = W
        # Create figure
        # Calculate nrows and ncols
        nrows = 2
        ncols = (len(group_name_list) + nrows - 1) // nrows
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 8))
        
        for i, (mean_salary, W) in enumerate(zip(mean_salary_list, W_list)):
            ax = axs[i//ncols, i%ncols]
            ax.plot(time_vals, mean_salary, c=self.colours["salary"])
            ax.set_title(fr"$W/N = {W:.0f}/{self.N:.0f}$", fontsize=8)
            ax.set(yscale="linear")
            ax.grid()

            # Axis labels. Only the bottom row should have x labels, and only the left column should have y labels
            subplot_spec = ax.get_subplotspec()
            if subplot_spec.is_last_row():
                ax.set_xlabel("Time")
            if subplot_spec.is_first_col():
                ax.set_ylabel("Price")
        
        
        # Text, save show,
        # If axs is 1d, then only index [0]
        if len(axs.shape) == 1:
            save_ax = axs[0]
        else:
            save_ax = axs[0, 0]
        
        self._text_save_show(fig, save_ax, "N_const_W_var", xtext=0.05, ytext=0.75, fontsize=6)

        
    def plot_salary_and_debt_distributions(self):
        """Plot the salary and debt distributions at the final time step
        """
        # Get data
        self._get_data(self.group_name)
        s = self.s[:, -1]
        d = self.d[:, -1]
        s_mean = np.mean(s)
        s_median = np.median(s)
        # Bins
        Nbins = int(np.sqrt(self.N))
        fig, (ax_s, ax_d) = plt.subplots(nrows=2)
        ax_s.axvline(x=s_mean, ls="--", c="k", label="Mean")
        ax_s.axvline(x=s_median, ls=":", c="k", label="Median")
        ax_s.hist(s, bins=Nbins, color=self.colours["salary"])
        ax_d.hist(d, bins=Nbins, color=self.colours["debt"])
        
        ax_s.set(xlabel="Salary", ylabel="Counts", yscale="log")
        ax_d.set(xlabel="Debt", yscale="log", ylabel="Counts")
        ax_s.set_title("Salary distribution", fontsize=8)
        ax_d.set_title("Debt distribution", fontsize=8)
        ax_s.grid() 
        ax_d.grid()
        self._add_legend(ax_s, ncols=2, fontsize=8)
        fig.suptitle("Final time distributions", fontsize=12)
        
        self._save_fig(fig, "salary_and_debt_distributions")
        plt.show()


    def plot_bankrupt_salary_mean_diff(self, time_vals: np.ndarray):
        """Distribution of the difference between the salary of companies that went bankrupt and the mean salary at times in time_vals
        """
        # Get data
        self._get_data(self.group_name)
        
        # Find the salary of companies that went bankrupt at the times in time_vals together with the mean salary
        bankrupt_salaries_diff_mean = []
        
        for t in time_vals:
            t_bankrupt_idx = self.went_bankrupt_idx[:, t]
            t_salary_bankrupt = self.s[t_bankrupt_idx, t-1]  # The interesting salary is at t-1
            mean_salary = np.mean(self.s[:, t-1])
            bankrupt_salary_mean_diff = t_salary_bankrupt - mean_salary
            bankrupt_salaries_diff_mean.append(bankrupt_salary_mean_diff)
        
        # Find the number of bins based on the length of the list of salary differences with the most elements
        Nbins = int(max([len(salaries) for salaries in bankrupt_salaries_diff_mean]))
        flattened_data = np.concatenate(bankrupt_salaries_diff_mean)
        logbins = np.geomspace(np.min(flattened_data), np.max(flattened_data), Nbins)
        
        # Determine ncols and nrows from the number of time_vals
        ncols = 2
        nrows = (len(time_vals) + ncols - 1) // ncols
        
        def _plotter(idx):
            ax = axs[idx//ncols, idx%ncols]
            ax.hist(bankrupt_salaries_diff_mean[idx], bins=logbins, color=self.colours["salary"])
            ax.set(xscale="symlog", yscale="linear")
            ax.set_title(f"Time = {time_vals[idx]}, N={len(bankrupt_salaries_diff_mean[idx])}", fontsize=8)
            ax.grid()
        
        fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(12, 8))
        for i in range(len(time_vals)):
            _plotter(i)
        
        # For the outer row and column, add x and y labels
        for ax in axs[-1]:
            ax.set_xlabel(r"Log $s_b - \mu$")
        for ax in axs[:, 0]:
            ax.set_ylabel("Counts")
            
        fig.suptitle("Salary difference between bankrupt companies and mean salary", fontsize=12)
        
        # Text, save, show
        self._text_save_show(fig, axs[0, 0], "bankrupt_salary_mean_diff", xtext=0.05, ytext=0.75)


    def plot_w0(self, time_vals: np.ndarray, no_bankrupt=False):
        """Plot the salary distribution of companies with w=0 for the times in time_vals

        Args:
            time_vals (np.ndarray): _description_
        """
        # Get data
        self._get_data(self.group_name)
        
        # Find the salary of companies that went bankrupt at the times in time_vals together with the mean salary
        s_w0 = []
        s_w0_no_bankrupt = []
        s_means = []
        s_companies_went_bankrupt = []
        
        for t in time_vals:
            w0 = self.w[:, t] == 0
            did_not_go_bankrupt = self.went_bankrupt_idx[:, t] == 0
            did_go_bankrupt = self.went_bankrupt_idx[:, t] == 1
            no_bankrupt_idx = np.logical_and(w0, did_not_go_bankrupt)
            s_w0_picked = self.s[w0, t]
            s_w0_no_bankrupt_picked = self.s[no_bankrupt_idx, t]
            salaries_bankrupt = self.s[did_go_bankrupt, t]
            s_mean = np.mean(self.s[:, t])  # Mean of ALL companies at time t
            
            # Store values
            s_w0.append(s_w0_picked)
            s_w0_no_bankrupt.append(s_w0_no_bankrupt_picked)
            s_companies_went_bankrupt.append(salaries_bankrupt)
            s_means.append(s_mean)

        # Find the number of bins based on the length of the list of salary differences with the most elements
        Nbins = int(np.sqrt(max([len(salaries) for salaries in s_w0])))
        flattened_data = np.concatenate(s_w0)
        logbins = np.geomspace(np.min(flattened_data), np.max(flattened_data), Nbins)
        
        # Determine ncols and nrows from the number of time_vals
        ncols = 2
        nrows = (len(time_vals) + ncols - 1) // ncols
        
        # Calculate bins and edges for plotting
        counts_list = []
        edges_list = []
        
        for i in range(len(time_vals)):
            counts, edges = np.histogram(s_w0[i], bins=logbins)
            counts_list.append(counts)
            edges_list.append(edges)

        # Get xlim and ylim
        ylim = (0, np.max(counts_list)+1)
        xlim = (logbins[0], np.max(a=(logbins[-1], np.max(np.concatenate(s_companies_went_bankrupt)))))
        
        def _plotter(idx):
            ax = axs[idx//ncols, idx%ncols]
            # Histograms
            if no_bankrupt:
                ax.hist(s_w0_no_bankrupt[idx], bins=logbins, color=self.colours["salary"], alpha=1, label=r"$s(w=0)$")
            else:
                ax.hist(edges_list[idx][:-1], edges_list[idx], weights=counts_list[idx], color=self.colours["salary"], alpha=1, label=r"$s(w=0)$")

            ax.hist(s_companies_went_bankrupt[idx], bins=logbins, color=self.colours["bankruptcy"], alpha=0.6, label=r"$s(w=0)$ bankrupt")
            # Means
            ax.axvline(x=s_means[idx], ls="--", c="k", alpha=0.8, label=r"Mean $s$")
            # Axis setup
            ax.set(xscale="log", yscale="linear", xlim=xlim, ylim=ylim)
            ax.set_title(f"Time = {time_vals[idx]}, N={len(s_w0[idx])}", fontsize=8)
            ax.grid()
            self._add_legend(ax, ncols=3, x=0.5, y=0.85, fontsize=7)
        
        fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(12, 8))
        for i in range(len(time_vals)):
            _plotter(i)
        
        # For the outer row and column, add x and y labels
        for ax in axs[-1]:
            ax.set_xlabel(r"Log salary")
        for ax in axs[:, 0]:
            ax.set_ylabel("Counts")
        
        if no_bankrupt:
            fig.suptitle(r"Salary of $w=0$ companies that did not go bankrupt", fontsize=12)
        else:
            fig.suptitle(r"Salary of $w=0$ companies", fontsize=12)
        
        # Text, save, show
        self._text_save_show(fig, axs[0, 0], "w0", xtext=0.05, ytext=0.75)


    def plot_all_salary(self):
        """Distribution of all salary values over time
        """
        # Get data
        self._get_data(self.group_name)
        # Bin it
        Nbins = int(np.sqrt(self.s.size) / 10)
        logbins = np.geomspace(self.s.min(), self.s.max(), Nbins)
        counts, edges = np.histogram(self.s, bins=logbins)
        
        # Plot
        fig, ax = plt.subplots()
        ax.hist(edges[:-1], edges, weights=counts, color=self.colours["salary"])
        ax.set(xlabel="Log Salary", ylabel="Log Counts", yscale="log", xscale="log", title="All salary values")
        ax.grid()
        
        # Text, save, show
        self._text_save_show(fig, ax, "all_salary")
        

    def plot_bankrupt_new_salary(self):
        """Plot the distribution salaries chosen by bankrupt companies over the whole time series, and below it all salaries.
        """
        # Get data and bin it
        self._get_data(self.group_name)
        # Skip values
        went_bankrupt_idx = self.went_bankrupt_idx[:, self.skip_values:]
        s = self.s[:, self.skip_values:]
        s_bankrupt = s[went_bankrupt_idx == 1]
        N_bankrupt = len(s_bankrupt)
        
        # Bins
        Nbins = int(np.sqrt(N_bankrupt))
        logbins = np.geomspace(np.min(self.s), np.max(s), Nbins)
        
        # Get y limits
        counts, edges = np.histogram(np.ravel(s), bins=logbins)
        ylim = (None, np.max(counts)*1.05)
        
        # Histogram
        fig, (ax, ax1) = plt.subplots(nrows=2)
        ax.hist(s_bankrupt, bins=logbins, color=self.colours["salary"])
        ax.set(ylabel="Log Counts", yscale="log", xscale="log", title="New salary of bankrupt companies", ylim=ylim)
        ax.grid()
        
        ax1.hist(edges[:-1], edges, weights=counts, color=self.colours["salary"])
        ax1.set(xlabel="Log Salary", ylabel="Log Counts", yscale="log", xscale="log", title="All salary values", ylim=ylim)
        ax1.grid()
        
        # Text, save, show
        self._text_save_show(fig, ax, "bankrupt_new_salary", fontsize=6)

    
    def plot_time_from_income_change_to_bankruptcy_distribution(self, show_plot=False):
        """Histogram of the time from income change to bankruptcy, together with a LLH fit to the distribution.
        """
        # Fit the data using log-likelihood minimization
        _, x_values, y_norm, y_lognorm, y_gamma, diff_vals = self.time_diff_llh_minimize(self.skip_values, show_plot)

        # Bin data for visualization 
        Nbins = int(0.3 * np.sqrt(len(diff_vals)))
        
        # Create figure
        fig, ax = plt.subplots()
        ax.hist(diff_vals, bins=Nbins, color=self.colours["time"], density=True, label="Data")      
        ax.plot(x_values, y_norm, c="grey", label="Double normal LLH fit")
        ax.plot(x_values, y_lognorm, c="k", label="Double Log-normal LLH fit")
        ax.plot(x_values, y_gamma, c="gray", label="Double Gamma LLH fit")
        ax.set(xlabel="Time", ylabel="Frequency", title="Time from income change to bankruptcy")  
        ax.grid()
        self._add_legend(ax, ncols=3)
        # Text, save, show
        self._text_save_show(fig, ax, "time_from_income_change_to_bankruptcy", xtext=0.05, ytext=0.85, fontsize=6)


    def plot_time_from_income_change_with_salary_debt_bankruptcy(self, line_start_mu1=None, line_start_mu2=None, time_steps_to_show=-1):
        """Plot the time from income change to bankruptcy as horizontal lines together with salary, debt and bankruptcy. 
        """        
        # Get data
        self._get_data(self.group_name)
        par, _, _, _ = self.time_diff_llh_minimize(self.skip_values)
        s_mean = np.mean(self.s, axis=0)
        d_mean = np.mean(self.d, axis=0)
        # Unpack parameters to get the mean of the two Gaussians
        mu1, mu2 = par[0], par[2]
        # Skip values
        if time_steps_to_show == -1:
            time_steps_to_show = self.time_steps
        s_mean = s_mean[self.skip_values: self.skip_values+time_steps_to_show]
        d_mean = d_mean[self.skip_values: self.skip_values+time_steps_to_show]
        self.went_bankrupt = self.went_bankrupt[self.skip_values: self.skip_values+time_steps_to_show] / self.N
        time_values = self.time_values[: time_steps_to_show]
        
        if np.any(line_start_mu1 == None):
            # x starts at the top point of the mean salary
            idx_line_start_mu1 = np.argmax(s_mean)
            x_line_start_mu1 = time_values[idx_line_start_mu1]
            y_line_start_mu1 = s_mean[idx_line_start_mu1]
            line_start_mu1 = (x_line_start_mu1, y_line_start_mu1)

        if np.any(line_start_mu2 == None):
            # x starts at the bottom point of the mean debt
            idx_line_start_mu2 = np.argmax(-d_mean)
            x_line_start_mu2 = time_values[idx_line_start_mu2]
            y_line_start_mu2 = d_mean[idx_line_start_mu2]
            line_start_mu2 = (x_line_start_mu2, y_line_start_mu2)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        twin_x_bankruptcy = ax.twinx()
        twin_x_d = ax.twinx()
        
        # Salary, debt, bankruptcy
        ax.plot(time_values, s_mean, label="Salary", c=self.colours["salary"])
        twin_x_d.plot(time_values, d_mean, label="Debt", c=self.colours["debt"])
        twin_x_bankruptcy.plot(time_values, self.went_bankrupt, label="Bankruptcy", c=self.colours["bankruptcy"], alpha=0.6)
        
        # Time diff
        # Horizontal
        ax.hlines(y=line_start_mu1[1], xmin=line_start_mu1[0], xmax=line_start_mu1[0]+mu1, ls="--", lw=3, colors="black", alpha=0.9)  # mu1, short time
        twin_x_d.hlines(y=line_start_mu2[1], xmin=line_start_mu2[0], xmax=line_start_mu2[0]+mu2, ls="-.", lw=3, colors="grey", alpha=0.9)  # mu2, long time, d min
        
        idx_min_s = np.argmin(s_mean)
        ax.hlines(y=0.0025+s_mean[idx_min_s], xmin=time_values[idx_min_s], xmax=time_values[idx_min_s] + mu2, ls="dotted", lw=3, colors="grey", alpha=0.9)  # mu2, long time, s min
        
        # Vertical
        ax.axvline(x=line_start_mu1[0]+mu1, ls="--", lw=2, color="black", alpha=0.9)  # mu1, short time
        twin_x_d.axvline(x=line_start_mu2[0]+mu2, ls="-.", lw=2, color="grey", alpha=0.9)  # mu2, long time, d min
        
        ax.axvline(x=time_values[idx_min_s] + mu2, ls="dotted", lw=2, color="grey", alpha=0.7)  # mu2, long time, s min
        
        # Axis setup        
        ax.set(xlabel="Time")
        ax.grid()
        ax.set_title("Salary, debt, bankruptcy and time from income change to bankruptcy", fontsize=10, )
        ax.tick_params(axis='y', labelcolor=self.colours["salary"])
        ax.set_ylabel("Mean Salary", color=self.colours["salary"])
        twin_x_d.tick_params(axis='y', labelcolor=self.colours["debt"])
        twin_x_bankruptcy.tick_params(axis='y', labelcolor=self.colours["bankruptcy"])
        twin_x_d.spines["right"].set_position(("axes", 1))

        # Text, save, show
        self._text_save_show(fig, ax, "time_diff_salary_debt_bankruptcy", xtext=0.05, ytext=0.85, fontsize=6)
    
    
    def plot_survivors(self, show_peak_plot=False):
        # Get data
        survive_arr = self.survive_bust_distribution(show_peak_plot)
        if len(survive_arr) == 0:
            print("No data")
            return
        # Number of bins
        max_survive = np.max(survive_arr)
        binwidth = int(np.floor(max_survive / 15))
        bins = np.arange(0, np.max(survive_arr)+1, binwidth)
        
        # Convert to fraction
        bins /= self.N
        survive_arr /= self.N
        
        # Create figure
        fig, ax = plt.subplots()
        ax.hist(survive_arr, bins=bins, color="grey")
        ax.set(xlabel="Survivors fraction", ylabel=f"Counts (bw={binwidth})", title="Fraction of companies that survive a bust")
        ax.grid()

        # # Add percentage values below
        # locs, labels = plt.xticks()
        # labels = ax.get_xticks()
        # sec = ax.secondary_xaxis(location=0)
        # fraction_values = np.array(labels) / self.N
        # sec.set_xticks(locs, labels=fraction_values)

        # Text, save, show
        self._text_save_show(fig, ax, "survivors", xtext=0.05, ytext=0.85, fontsize=6)
    
    
    def plot_running_KDE(self, bandwidth_s=None, bandwidth_d=None, eval_points=100, s_lim=None, d_lim=None, kernel="gaussian", show_mean=False, show_title=False, plot_debt=True):
        # Get data
        s_eval, KDE_prob = self.running_KDE("salary", bandwidth_s, eval_points, kernel, s_lim)  # KDE probabilities
        d_eval, KDE_prob_d = self.running_KDE("capital", bandwidth_d, eval_points, kernel, d_lim)  # KDE probabilities
        time_values, mu, d_mean = self._skip_values(self.time_values, self.mu, self.d.mean(axis=0))  # Get mean salary and skip values
        
        # Create figure
        figsize = (16, 12) if plot_debt else (10, 5)
        ncols, nrows = (1, 2) if plot_debt else (1, 1)
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        if plot_debt:
            ax, ax_d = ax
        
        # Title
        if bandwidth_s is None:
            bandwidth_s = s_eval[1] - s_eval[0]
        if bandwidth_d is None:
            bandwidth_d = d_eval[1] - d_eval[0]
        title = f"KDE, bw salary={bandwidth_s}, bw capacity={bandwidth_d}"
        if show_title: fig.suptitle(title, fontsize=10)
    
        # Salary
        label_fontsize = 15
        ticks_fontsize = 10
        im = ax.imshow(KDE_prob, aspect="auto", origin="lower", extent=[self.skip_values, self.time_steps, np.min(s_eval), np.max(s_eval)], cmap="hot")
        if show_mean: ax.plot(time_values, mu/self.W, c="magenta", label=r"$\mu / W$", alpha=1, lw=1)
        ax.set_xlabel("Time", fontsize=label_fontsize)
        ax.set_ylabel("Salary", fontsize=label_fontsize)
        ax.tick_params(axis='both', which='major', labelsize=ticks_fontsize)
        
        # Debt
        if plot_debt:
            im_d = ax_d.imshow(KDE_prob_d, aspect="auto", origin="lower", extent=[self.skip_values, self.time_steps, np.min(d_eval), np.max(d_eval)], cmap="magma")
            if show_mean: ax_d.plot(time_values, -d_mean, c="red", label="Mean capacity")
            ax_d.set_ylabel("Capacity", fontsize=label_fontsize)
            ax_d.set_xlabel("Time", fontsize=label_fontsize)
            ax_d.tick_params(axis='both', which='major', labelsize=ticks_fontsize)
            
            ax.set(xticks=[], xlabel="")
            
        
        # Title and axis setup
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, ticks=[], pad=0.01)
        cbar.set_label(label="Frequency", fontsize=label_fontsize)
        
        if plot_debt:
            cbar_d = fig.colorbar(im_d, ax=ax_d, ticks=[], pad=0.01)
            cbar_d.set_label(label="Frequency", fontsize=label_fontsize)

        # Text, save, show
        self._text_save_show(fig, ax, "running_KDE", xtext=0.05, ytext=0.95, fontsize=0)
    

    def animate_w0_wnon0(self, skip_time_steps=0):
        """Make a histogram animation of the salary distribution of companies with w=0 at each time step together with the salary picked of bankrupt companies, and another with w>0
        """
        # Load data and exclude warmup
        self._get_data(self.group_name)

        # Find the salary of companies that went bankrupt at the times in time_vals together with the mean salary
        s_wnon0 = []
        s_w0_no_bankrupt = []
        s_means = []
        s_medians = []
        s_companies_went_bankrupt = []
        
        frames_to_play = np.arange(self.skip_values, self.time_steps)[::skip_time_steps]
        
        # Calculate the bin counts for the histogram
        
        for t in frames_to_play:
            w_non0 = self.w[:, t] > 0
            w0 = self.w[:, t] == 0
            did_not_go_bankrupt = self.went_bankrupt_idx[:, t] == 0
            did_go_bankrupt = self.went_bankrupt_idx[:, t] == 1
            no_bankrupt_idx = np.logical_and(w0, did_not_go_bankrupt)
            s_w0_no_bankrupt_picked = self.s[no_bankrupt_idx, t]
            salaries_bankrupt = self.s[did_go_bankrupt, t]
            s_mean = np.mean(self.s[:, t])  # Mean of ALL companies at time t
            s_median = np.median(self.s[:, t])
            s_non0 = self.s[w_non0, t]
            # Store values
            s_wnon0.append(s_non0)
            s_w0_no_bankrupt.append(s_w0_no_bankrupt_picked)
            s_companies_went_bankrupt.append(salaries_bankrupt)
            s_means.append(s_mean)
            s_medians.append(s_median)

        # Find the number of bins based on the length of the list of salary differences with the most elements
        # The bins edges go from the minimum value to the maximum value of the salaries. The w=0 companies have lower salaries and w>0 higher
        Nbins = int(np.sqrt(max([len(salaries) for salaries in s_wnon0])))
        flattened_data_for_max = np.concatenate(s_wnon0)
        logbins = np.geomspace(np.min(self.s), np.max(flattened_data_for_max), Nbins)  
        
        # Get bin counts
        s_wnon0_hist = []
        s_w0_no_bankrupt_hist = []
        s_companies_went_bankrupt_hist = []
        
        for _ in range(len(frames_to_play)):
            s_wnon0_hist.append(np.histogram(s_non0, bins=logbins)[0])
            s_w0_no_bankrupt_hist.append(np.histogram(s_w0_no_bankrupt_picked, bins=logbins)[0])
            s_companies_went_bankrupt_hist.append(np.histogram(salaries_bankrupt, bins=logbins)[0])

        # Get ylim from the min and max of counts
        ylim = (0, 2*np.max([np.max(s_wnon0_hist), np.max(s_w0_no_bankrupt_hist)]))  # 10% to add a small margin
        
        # Figure setup        
        fig, (ax_0, ax_non0) = plt.subplots(nrows=2)
        _, _, s_bar_container = ax_0.hist(s_w0_no_bankrupt[0], bins=logbins, color=self.colours["salary"])  # Initial histogram 
        _, _, bankrupt_bar_container = ax_0.hist(s_companies_went_bankrupt[0], bins=logbins, color=self.colours["bankruptcy"], alpha=0.6)  # Initial histogram
        vline_0 = ax_0.axvline(x=s_means[0], ls="--", c="k", alpha=0.8, label="Mean")
        median_line_0 = ax_0.axvline(x=s_medians[0], ls="dotted", c="grey", alpha=0.8, label="Median")
        ax_0.set_yscale("symlog")  
        ax_0.set(xlim=(logbins[0], logbins[-1]), ylim=ylim, ylabel="Counts", xscale="log", title="w=0")
        ax_0.grid()
        
        _, _, s_non0_bar_container = ax_non0.hist(s_wnon0[0], bins=logbins, color=self.colours["salary"])  # Initial histogram 
        vline_non0 = ax_non0.axvline(x=s_means[0], ls="--", c="k", alpha=0.8, label="Mean")
        median_line_non0 = ax_non0.axvline(x=s_medians[0], ls="dotted", c="grey", alpha=0.8, label="Median")
        ax_non0.set(xlim=(logbins[0], logbins[-1]), ylim=ylim, xlabel=r"Workers $w$", ylabel="Counts", xscale="log", yscale="symlog", title="w>0")
        ax_non0.grid()
        
        def _add_text_and_title(idx):            
            """ Set the title to the current time step.
            Display the mean salary, the number of companies at w=0 and the number of bankrupt companies
            """
            fig.suptitle(f"Time = {frames_to_play[idx]}")
            text_to_display = f"Mean salary = {s_means[idx]:.1e}\nN(w=0) non-bankrupt = {len(s_w0_no_bankrupt[idx])}\nN(w=0) bankrupt = {len(s_companies_went_bankrupt[idx])}"
            ax_0.text(0.95, 0.85, text_to_display, transform=ax_0.transAxes, bbox=dict(facecolor='darkgray', alpha=0.5), horizontalalignment='right')
        
        # Text
        _add_text_and_title(idx=0)
        if self.add_parameter_text_to_plot: self._add_parameters_text(ax_0, fontsize=6, y=0.85)

        def animate(i, s_bar_container, bankrupt_bar_container, s_non0_bar_container):
            """Frame animation function for creating a histogram."""
            # s no bankrupt histogram
            s_data = s_w0_no_bankrupt[i]
            n_s, _ = np.histogram(s_data, logbins)
            for count_s, rect_s in zip(n_s, s_bar_container.patches):
                rect_s.set_height(count_s)

            # Bankrupt histogram
            data_bankrupt = s_companies_went_bankrupt[i]
            n_w, _ = np.histogram(data_bankrupt, logbins)
            for count, rect in zip(n_w, bankrupt_bar_container.patches):
                rect.set_height(count)
                
            # w>0 histogram
            s_non0_data = s_wnon0[i]
            n_non0, _ = np.histogram(s_non0_data, logbins)
            for count_non0, rect_non0 in zip(n_non0, s_non0_bar_container.patches):
                rect_non0.set_height(count_non0)

            # Update the mean vlines
            vline_0.set_xdata([s_means[i], s_means[i]])
            vline_non0.set_xdata([s_means[i], s_means[i]])
            median_line_0.set_xdata([s_medians[i], s_medians[i]])
            median_line_non0.set_xdata([s_medians[i], s_medians[i]])

            # Set the title and text
            _add_text_and_title(i)
                        
            return s_bar_container.patches + bankrupt_bar_container.patches + s_non0_bar_container.patches + [vline_0, vline_non0, median_line_0, median_line_non0]
        
        # Create the animation
        anim = functools.partial(animate, s_bar_container=s_bar_container, bankrupt_bar_container=bankrupt_bar_container, s_non0_bar_container=s_non0_bar_container)  # Necessary when making histogram
        ani = animation.FuncAnimation(fig, anim, frames=len(frames_to_play), interval=1)
        
        # Save animation
        self._save_anim(ani, name="w0_wnon0_anim")        
        
    
    def plot_diversity(self):
        # Load data
        time_vals, diversity = self._worker_diversity()
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(time_vals, diversity, c=self.colours["diversity"])
        ax.set(xlabel="Time", ylabel="Diversity", ylim=(0, self.N), xlim=self.xlim)
        ax.grid()
        
        # Text, save, show
        self._text_save_show(fig, ax, "diversity", xtext=0.05, ytext=0.85, fontsize=6)
        
        
    def plot_diversity_multiple_alpha(self, group_name_list):
        """Plot the worker diversity for different alpha values, each on their own subplot
        """       
        # Load data
        diversity_list = []
        alpha_list = []
        mean_s_list = []
        for gname in group_name_list:
            # Get values
            self._get_data(gname)
            time_vals, diversity = self._worker_diversity(gname)
            s = self._skip_values(self.s)
            mean_s = np.mean(s, axis=0)
            # Append values
            diversity_list.append(diversity)
            alpha_list.append(self.prob_expo)
            mean_s_list.append(mean_s)
        
        s_min = np.min([np.min(s) for s in mean_s_list])
        s_max = np.max([np.max(s) for s in mean_s_list])
        s_ylim = (s_min, s_max)
        
        # Create figure
        nrows = 2
        ncols = (len(group_name_list) + nrows - 1) // nrows
        fig, axs = plt.subplots(figsize=(10, 5), ncols=ncols, nrows=nrows)
        
        # Loop over axes
        for i in range(len(group_name_list)):
            # Create twin axis where the mean salary is plotted            
            ax = axs[i//ncols, i%ncols]
            twinx = ax.twinx()
            twinx.plot(time_vals, mean_s_list[i], c=self.colours["salary"], alpha=0.9)
            twinx.set(ylim=s_ylim)
            
            ax.plot(time_vals, diversity_list[i], c=self.colours["diversity"])
            ax.set(ylim=(0, self.N))
            ax.set_title(fr"$\alpha = {alpha_list[i]:.0f}$")
            ax.grid()
            
            # Axis labels
            subplot_spec = ax.get_subplotspec()
            if subplot_spec.is_last_row():
                ax.set_xlabel("Time")
            else:
                ax.set_xticklabels([])
                twinx.set_xticklabels([])
            if subplot_spec.is_first_col():
                ax.set_ylabel("Diversity", color=self.colours["diversity"])
                ax.tick_params(axis='y', labelcolor=self.colours["diversity"])
                twinx.set_yticklabels([])
            elif subplot_spec.is_last_col():
                twinx.tick_params(axis='y', labelcolor=self.colours["salary"])
                twinx.set_ylabel("Mean salary", color=self.colours["salary"])                
                ax.set_yticklabels([])
            else:
                ax.set_yticklabels([])
                twinx.set_yticklabels([])

        # Text save show
        self._text_save_show(fig, axs[0, 0], "diversity_multiple_alpha", xtext=0.05, ytext=0.85, fontsize=6)
        
        
    def plot_running_KDE_multiple_s_min(self, group_name_list, bandwidth_s=None, eval_points=100, s_lim=None, kernel="gaussian", show_mean=False, show_title=False):
        figsize = (10, 5)
        less_than_four_datasets = len(group_name_list) < 4
        if less_than_four_datasets:
            nrows = 1
            ncols = len(group_name_list)
        else:
            nrows = 2
            ncols = (len(group_name_list) + nrows - 1) // nrows
        
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

        label_fontsize = 15
        ticks_fontsize = 10
        
        self._get_data(group_name_list[0])
        s_eval_arr = np.zeros((len(group_name_list), eval_points))
        KDE_prob_arr = np.zeros((len(group_name_list), eval_points, self.time_steps-self.skip_values))
        mu_arr = np.zeros((len(group_name_list), self.time_steps-self.skip_values))
        s_min_arr = np.zeros(len(group_name_list))

        def _get_KDE(gname):
            # Load data
            self._get_data(gname)
            # KDE
            s_eval, KDE_prob = self.running_KDE("salary", bandwidth_s, eval_points, kernel, s_lim, gname=gname)  # KDE probabilities
            time_values, mu = self._skip_values(self.time_values, self.mu)  # Get mean salary and skip values
            s_min = self.s_min
            return s_eval, KDE_prob, time_values, mu, s_min

        # Calculate data
        for i, gname in enumerate(group_name_list):
            s, KDE, time_values, mu, s_min = _get_KDE(gname)

            s_eval_arr[i, :] = s
            KDE_prob_arr[i, :, :] = KDE
            mu_arr[i, :] = mu
            s_min_arr[i] = s_min
            

        # Calculate minimum and maximum values for the colorbar
        min_val = np.min(KDE_prob_arr)
        max_val = np.max(KDE_prob_arr)

        # Tick setup
        x_tick_first, x_tick_last = self.skip_values+50, self.time_steps-50
        x_ticks = [x_tick_first, x_tick_first+(x_tick_last-x_tick_first)/2, x_tick_last]
        x_tick_labels = x_ticks
        y_ticks = [s_lim[0], s_lim[0]+(s_lim[1]-s_lim[0])/2, s_lim[1]]
        y_tick_labels = [s_lim[0], s_lim[1]]
        
        # Run the plotting function
        for index in range(len(group_name_list)):
            # Salary
            
            if len(group_name_list) == 1:
                ax = axs
            elif less_than_four_datasets:
                ax = axs[index]
            else:
                ax = axs[index//ncols, index%ncols]
                
            im = ax.imshow(KDE_prob_arr[index], aspect="auto", origin="lower",
                        extent=[self.skip_values, self.time_steps, np.min(s_eval_arr[index]), np.max(s_eval_arr[index])],
                        cmap="hot", vmin=min_val, vmax=max_val)
            if show_mean:
                ax.plot(time_values, mu_arr[index] / self.W, c="magenta", label=r"$\mu / W$", alpha=1, lw=0.6)

            # Axis and ticks
            self._axis_ticks_and_labels(ax, x_ticks, y_ticks, x_tick_labels, y_tick_labels, x_dtype="int")
            self._axis_labels_outer(ax, x_label="Time", y_label="Wage")

        # Add a single colorbar for all subplots, stretch it to the full height of the figure
        cbar = fig.colorbar(im, ax=axs, orientation='vertical', ticks=[], pad=0.01)# fraction=0.02, pad=0.04)
        cbar.set_label(label="Frequency", fontsize=label_fontsize)
        cbar.ax.tick_params(labelsize=0)  # Remove tick labels

        if len(group_name_list) == 1:
            save_ax = axs
        elif less_than_four_datasets:
            save_ax = axs[0]
        else:
            save_ax = axs[0, 0]

        # Text, save, show
        self._text_save_show(fig, save_ax, "running_KDE_multiple_s_min", xtext=0.05, ytext=0.95, fontsize=0)
        
        
    def plot_return_distribution(self, data_name: str, yscale, Nbins=None, ylim=None):
        """Plot the distribution of relative change in mu i.e. the return of the asset price.
        """
        # Load data
        self._get_data(self.group_name)
        
        # Return
        r = self._asset_return(data_name, time_period=1)
        
        # Bin data
        if Nbins is None:
            Nbins = int(0.5 * np.sqrt(len(r)))   # A large alpha creates larger variance, thus the need for wider bins
        bins = np.linspace(np.min(r), np.max(r), Nbins)
        counts, edges = np.histogram(r, bins=bins)
        bin_width = edges[1] - edges[0]

        # LLH fit to the distribution
        x_fit, y_gauss, y_student, y_lst = self._return_fit(r)

        # Create figure
        fig, ax_hist = plt.subplots(figsize=(10, 5))
        
        # Disitribution of returns
        ax_hist.hist(edges[:-1], edges, weights=counts, color=self.colours["mu"], label="Data", density=True)
        # Fits
        ax_hist.plot(x_fit, y_gauss, c="k", label="Gaussian fit")
        # # ax_hist.plot(x_fit, y_student, c="grey", label="Student's t fit")
        ax_hist.plot(x_fit, y_lst, c="gray", label="Loc. trans. Student t")

        # Axis setup
        if ylim is None:
            counts_min = 1e-4
            counts_max = 1e1
            ylim = (counts_min, counts_max)
        ax_hist.set(xlabel="Return", ylabel=f"Probability density (bw={bin_width:.2f})", yscale=yscale, ylim=ylim)
        ax_hist.grid()
        self._add_legend(ax_hist, ncols=3, y=0.9)
        
        # Text, save, show
        self._text_save_show(fig, ax_hist, "return", xtext=0.05, ytext=0.85, fontsize=6)
                
                
    def plot_autocorr(self, data_name:str, time_period=1, max_lag=None, same_subplot=False):
        # Get data and calculate autocorrelation
        self._get_data(self.group_name)
        lag_vals = np.arange(max_lag+1)  # Include endpoint
        r = self._asset_return(data_name, time_period)
        autocorr = general_functions.sample_autocorrelation(r, max_lag=max_lag)
        r2 = np.abs(r)#r ** 2
        autocorr_r2 = general_functions.sample_autocorrelation(r2, max_lag=max_lag)
        
        # Fit a power law to the r2 autocorrelation
        
        def _power_law(x, a, b):
            return a * x ** (-b)
        
        par, cov = scipy.optimize.curve_fit(_power_law, lag_vals[1:], autocorr_r2[1:])  # Skip the first value because it is 1 
        x_fit = np.linspace(1, max_lag, 100)
        y_fit = _power_law(x_fit, *par)
        
        # Create figure
        title_r1 = fr"$r_{{\tau = {time_period}}}(t)$"
        title_r2 = fr"$|r_{{\tau = {time_period}}}(t)|$"
        
        # Color
        if data_name == "mu":
            colour = self.colours["mu"]
        elif data_name == "capital_individual_mean":
            colour = self.colours["capital"]

        if same_subplot:
            nrows = 1
        else:
            nrows = 2
            
        fig, axs = plt.subplots(nrows=nrows, figsize=(10, 5))
        if same_subplot:
            ax = axs

            # r
            ax.plot(lag_vals, autocorr, marker="o", ls="--", c=colour, lw=1, label=title_r1)
            ax.grid()
            
            # |r|
            ax.plot(lag_vals, autocorr_r2, "-*", c="k", label=title_r2, lw=1, markersize=9)
            # ax.plot(x_fit, y_fit, label=f"Power law fit, a={par[1]:.2f}", c="k", lw=1, label=title_r2)
            ax.set(xlabel="Lag", ylabel="Autocorrelation")
            # Ticks
            # 6 y ticks, only 0 and 1 has label
            # 6 x ticks, only 0, 15, and 30 has label
            x_ticks = np.linspace(0, 30, 7)
            y_ticks = np.linspace(0, 1, 5)
            x_label = [0, 15, 30]
            y_label = [0, 1]
            self._axis_ticks_and_labels(ax, x_ticks, y_ticks, x_label, y_label)
            self._add_legend(ax, x=0.5, y=0.88, ncols=2, fontsize=13)

        # Different subplots
        else:        
            ax, ax_r2 = axs
            
            ax.plot(lag_vals, autocorr, "--x", c=colour, lw=1)
            ax.set(ylabel="Autocorrelation", title=title_r1)
            ax.grid()
            
            # r^2
            ax_r2.plot(lag_vals, autocorr_r2, "--x", c=colour, label="Data")
            ax_r2.plot(x_fit, y_fit, label=f"Power law fit, a={par[1]:.2f}", c="k", lw=1)
            ax_r2.set(xlabel="Lag", ylabel="Autocorrelation", title=title_r2)
            ax_r2.grid()
            
            self._add_legend(ax_r2, x=0.5, y=0.7, ncols=2, fontsize=15)
        
        # Text, save, show
        self._text_save_show(fig, ax, "autocorr", xtext=0.05, ytext=0.85, fontsize=6)
        

    def plot_mu_return_different_time(self, time_period_list, data_name:str, yscale, Nbins=None, gauss_std=0.5):
        """Calculate and plot the return for different time periods in the asset return calculation, together with a standard Gaussian.
        """
        # Load data
        self._get_data(self.group_name)    
        
        number_of_elements = self._skip_values(self.time_steps)
        
        # Get bins
        if Nbins is None:
            Nbins = int(0.5 * np.sqrt(number_of_elements))  # A large alpha creates larger variance, thus the need for wider bins
        marker_list = general_functions.list_of_markers
        
        counts_min = [10]
        counts_max = [0]
        
        def plot_r(axis, i):
            # Calculate r and normalize it
            r = self._asset_return(data_name, time_period_list[i])
            counts, edges = np.histogram(r, bins=Nbins, density=True)
            # Plot
            axis.plot(edges[:-1], counts, marker=marker_list[i], markersize=10, label=fr"$\tau = $ {time_period_list[i]}", lw=1)
            # Update min and max counts
            counts_min[0] = min(counts_min[0], np.min(counts[counts>1e-18]))
            counts_max[0] = max(counts_max[0], np.max(counts))
        
        # Create figure
        fig, ax_hist = plt.subplots(figsize=(10, 5))
        # Plot the Gaussian
        gauss_mu = 0
        x_gauss = np.linspace(-4*gauss_std, 4*gauss_std, 200)  # 3 standard ddviations
        y_gauss = scipy.stats.norm.pdf(x_gauss, loc=gauss_mu, scale=gauss_std)
        # Plot the return values
        ax_hist.plot(x_gauss, y_gauss, label="Gaussian", lw=2)
        for i in range(len(time_period_list)):
            plot_r(ax_hist, i)

        # Axis setup
        ax_hist.set(xlabel="Return", ylabel=f"Probability density", yscale=yscale, ylim=(counts_min[0]*0.9, counts_max[0]*1.1))
        ax_hist.grid()
        self._add_legend(ax_hist, ncols=len(time_period_list)+1)
        
        # Text, save, show
        self._text_save_show(fig, ax_hist, "return_different_time", xtext=0.05, ytext=0.85, fontsize=6)
        

    def plot_running_KDE_NW(self, group_name_list, bandwidth=None, eval_points=100, lim=None, kernel="gaussian", show_mean=False, show_title=False):
        # Create figure
        less_than_four_datasets = len(group_name_list) < 4
        if less_than_four_datasets:
            ncols = 1
            nrows = len(group_name_list)
        else:
            nrows = 2
            ncols = (len(group_name_list) + nrows - 1) // nrows
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 8))

        # Get data
        self._get_data(group_name_list[0])
        s_eval_arr = np.zeros((len(group_name_list), eval_points))
        KDE_prob_arr = np.zeros((len(group_name_list), eval_points, self.time_steps-self.skip_values))
        mu_arr = np.zeros((len(group_name_list), self.time_steps-self.skip_values))
        N_arr = np.zeros(len(group_name_list))
        W_arr = np.zeros_like(N_arr)

        title_fontsize = 15

        def _get_KDE(gname):
            # Load data
            self._get_data(gname)
            # KDE
            s_eval, KDE_prob = self.running_KDE("salary", bandwidth, eval_points, kernel, lim, gname=gname)  # KDE probabilities
            time_values, mu = self._skip_values(self.time_values, self.mu)  # Get mean salary and skip values
            N = self.N
            W = self.W
            return s_eval, KDE_prob, time_values, mu, N, W

        # Calculate data
        for i, gname in enumerate(group_name_list):
            s, KDE, time_values, mu, N, W = _get_KDE(gname)

            s_eval_arr[i, :] = s
            KDE_prob_arr[i, :, :] = KDE
            mu_arr[i, :] = mu
            N_arr[i] = N
            W_arr[i] = W

        # Calculate minimum and maximum values for the colorbar
        min_val = np.min(KDE_prob_arr)
        max_val = np.max(KDE_prob_arr)

        # Initialize im to None
        im = None

        def _plot_KDE(index):
            nonlocal im  # Declare im as nonlocal to update it within the function
            # Get current axis
            if len(group_name_list) == 1:
                ax = axs
            elif less_than_four_datasets:
                ax = axs[index]
            else:
                ax = axs[index // ncols, index % ncols]
                            
            im = ax.imshow(KDE_prob_arr[index], aspect="auto", origin="lower",
                        extent=[self.skip_values, self.time_steps, np.min(s_eval_arr[index]), np.max(s_eval_arr[index])],
                        cmap="hot", vmin=min_val, vmax=max_val)
            if show_mean:
                ax.plot(time_values, mu_arr[index] / W_arr[index], c="magenta", label=r"$\mu / W$", alpha=1, lw=0.6)
            ax.tick_params(axis='both', which='major')

            # Title
            title = fr"$N =$ {N_arr[index]:.0f}, $W =$ {W_arr[index]:.0f}"
            ax.set_title(title, fontsize=title_fontsize)

            # Axis labels. Only the bottom row should have x labels, and only the left column should have y labels
            # Also remove the ticks of inner subplots
            subplot_spec = ax.get_subplotspec()
            if subplot_spec.is_last_row():
                ax.set_xlabel("Time")
            else:
                ax.set_xticks([])
            if subplot_spec.is_first_col():
                ax.set_ylabel("Wage")
            else:
                ax.set_yticks([])

        # Run the plotting function
        for i in range(len(group_name_list)):
            _plot_KDE(i)

        # Add a single colorbar for all subplots, stretch it to the full height of the figure
        cbar = fig.colorbar(im, ax=axs, orientation='vertical')# fraction=0.02, pad=0.04)
        cbar.set_label(label="Frequency", fontsize=15)
        cbar.ax.tick_params(labelsize=0)  # Remove tick labels

        # Text, save, show
        if len(group_name_list) == 1:
            save_ax = axs
        elif less_than_four_datasets:
            save_ax = axs[0]
        else:
            save_ax = axs[0, 0]
        self._text_save_show(fig, save_ax, "running_KDE_NW", xtext=0.05, ytext=0.95, fontsize=0)
        
        
    def plot_mu_capital_sum_and_individual(self, Nbins=None, show_time_series=True, show_distributions=True, ylim_pdf=None):
        """Two subfigures, one with mean capital and mu, the other with the return distributions using mu, individual capital and sum of capital
        """
        assert show_time_series or show_distributions, "At least one of the subfigures must be shown"
        
        # Get data
        self._get_data(self.group_name)
        # Skip values
        time_values, mu, d = self._skip_values(self.time_values, self.mu, self.d)
        capital = -d
        capital_mean = np.mean(capital, axis=0)

        if show_distributions:
            # For the distributions, we need to first get the bins and then bin the data
            # Return distributions
            time_period = 1
            r_mu = self._asset_return("mu", time_period)
            r_capital_individual = self._asset_return("capital_individual", time_period)
            r_capital_sum = self._asset_return("capital_sum", time_period)

            # Get bins
            if Nbins is None:
                Nbins = int(0.9 * np.sqrt(len(r_mu)))
            r_min = np.min((r_mu.min(), r_capital_individual.min(), r_capital_sum.min()))
            r_max = np.max((r_mu.max(), r_capital_individual.max(), r_capital_sum.max()))
            bins = np.linspace(r_min, r_max, Nbins)
            bin_width = bins[1] - bins[0]
            # Bin the data
            counts_mu, _ = np.histogram(r_mu, bins=bins, density=True)
            counts_capital_individual, _ = np.histogram(r_capital_individual, bins=bins, density=True)
            counts_capital_sum, _ = np.histogram(r_capital_sum, bins=bins, density=True)
            # y limits for the return distributions
            if ylim_pdf is None:
                ylim_pdf = (1e-4, 1)

        # Create figure
        show_both = show_time_series and show_distributions
        if show_both:
            nrows = 5
        elif show_distributions:
            nrows = 4
        else:
            nrows = 1
        fig, axs = plt.subplots(nrows=nrows, figsize=(10, 5))
        if show_both:
            ax_time, ax_return = axs
            save_ax = ax_time
        elif show_distributions:
            ax_return = axs
            save_ax = ax_return
        else:
            ax_time = axs
            save_ax = ax_time
        
        if show_time_series:
            # Time series
            # Need to have capital and mu on shared x axis
            twinx = ax_time.twinx()
            twinx.plot(time_values, capital_mean, label="Mean capital", c=self.colours["debt"])
            twinx.tick_params(axis='y', labelcolor=self.colours["debt"])
            twinx.set_ylabel("Capital", color=self.colours["debt"])

            ax_time.plot(time_values, mu / self.W, label=r"$\mu/W$", c=self.colours["mu"])
            ax_time.set(xlabel="Time", ylabel="Price", title=r"Mean capital and $\mu$")
            ax_time.grid()
            ax_time.tick_params(axis="y", labelcolor=self.colours["mu"])    
            ax_time.set_ylabel(r"$\mu/W$", color=self.colours["mu"])

        if show_distributions:        
            # Return distributions
            ax_return.hist(bins[:-1], bins, weights=counts_mu, color=self.colours["mu"], label=r"$\mu$", alpha=0.95, density=True)
            ax_return.hist(bins[:-1], bins, weights=counts_capital_individual, color=self.colours["capital"], label=r"Individual $C$", alpha=0.8, density=True)
            ax_return.hist(bins[:-1], bins, weights=counts_capital_sum, color=self.colours["debt"], label=r"Mean $C$", alpha=0.6, density=True)
            ax_return.set(xlabel="Return", ylabel=f"Prob. Density", title="Return distributions", yscale="log")
            ax_return.grid()
            self._add_legend(ax_return, ncols=3, x=0.3, y=0.8)   
        
        # Text, save, show
        self._text_save_show(fig, save_ax, "mu_capital_sum_and_individual", xtext=0.05, ytext=0.85, fontsize=6)
        
        
    def plot_different_return_distribution_definitions(self, Nbins=None, Nbins_indi=None, ylim=None):
        """The four different definitions for returns. 

        Args:
            Nbins (_type_, optional): _description_. Defaults to None.
            Nbins_indi (_type_, optional): _description_. Defaults to None.
            ylim (_type_, optional): _description_. Defaults to None.
        """
        # Get data
        self._get_data(self.group_name)

        # For the distributions, we need to first get the bins and then bin the data
        # Return distributions
        time_period = 1
        r_mu = self._asset_return("mu", time_period)
        r_capital_individual = self._asset_return("capital_individual_mean", time_period)
        r_capital_sum = self._asset_return("capital_sum", time_period)
        r_capital_individual_all = self._asset_return("capital_individual_all", time_period)

        # Get bins
        if Nbins is None:
            Nbins = int(0.9 * np.sqrt(len(r_mu)))
        if Nbins_indi is None:
            Nbins_indi = int(0.9 * np.sqrt(len(r_capital_individual_all)))
        r_min = np.min((r_mu.min(), r_capital_individual.min(), r_capital_sum.min()))
        r_max = np.max((r_mu.max(), r_capital_individual.max(), r_capital_sum.max()))
        bins = np.linspace(r_min, r_max, Nbins)
                        
        # Bin the data
        counts_mu, _ = np.histogram(r_mu, bins=bins, density=True)
        counts_capital_individual, _ = np.histogram(r_capital_individual, bins=bins, density=True)
        counts_capital_sum, _ = np.histogram(r_capital_sum, bins=bins, density=True)
        
        # y limits for the return distributions
        if ylim is None:
            ylim = (5e-4, 1)

        # Create figure
        fig, axs = plt.subplots(figsize=(10, 5), nrows=2, ncols=1)
        ax_aggr, ax_indi = axs
        
        # ax_aggr: Aggregate distributions (mu, mu mean, mu individual mean)
        ax_aggr.hist(bins[:-1], bins, weights=counts_mu, color=self.colours["mu"], label=r"$\mu$", alpha=0.95, density=True)
        ax_aggr.hist(bins[:-1], bins, weights=counts_capital_individual, color=self.colours["capital"], label=r"Mean Individual $C$", alpha=0.8, density=True)
        ax_aggr.hist(bins[:-1], bins, weights=counts_capital_sum, color=self.colours["debt"], label=r"Mean $C$", alpha=0.6, density=True)
        ax_aggr.set(ylabel=f"Prob. Density", yscale="log", ylim=ylim, title="Distribution of Aggregate changes")
        ax_aggr.grid()
        self._add_legend(ax_aggr, ncols=3, x=0.3, y=0.8)   
        
        # ax_indi: Individual capital distributions
        ax_indi.hist(r_capital_individual_all, bins=Nbins_indi, color=self.colours["capital"], label=r"Individual $C$", alpha=1, density=True)
        ax_indi.set(xlabel="Return", yscale="log", ylim=ylim, ylabel="Prob. Density", title=r"Distribution of individual $C$ changes")
        ax_indi.grid()
        
        # Text, save, show
        self._text_save_show(fig, ax_aggr, "mu_capital_sum_and_individual", xtext=0.05, ytext=0.85, fontsize=6)


    def plot_return_individual_and_aggregate(self, Nbins_agg=None, Nbins_indi=None, ylim=None):
        # Get data
        self._get_data(self.group_name)

        # For the distributions, we need to first get the bins and then bin the data
        # Return distributions
        time_period = 1
        r_agg = self._asset_return("capital_individual_mean", time_period)
        r_indi = self._asset_return("capital_individual_all", time_period)

        # Get bins
        if Nbins_agg is None:
            Nbins_agg = int(0.9 * np.sqrt(len(r_agg)))
        if Nbins_indi is None:
            Nbins_indi = int(0.9 * np.sqrt(len(r_indi)))
        indi_bins = np.linspace(r_indi.min(), r_indi.max(), Nbins_indi)
        agg_bins = np.linspace(r_agg.min(), r_agg.max(), Nbins_agg)
                        
        # Bin the data
        counts_agg, edges_agg = np.histogram(r_agg, bins=agg_bins, density=True)
        counts_indi, edges_indi = np.histogram(r_indi, bins=indi_bins, density=True)
        
        # y limits for the return distributions
        if ylim is None:
            ylim = (5e-4, 1)

        # Create figure
        fig, axs = plt.subplots(figsize=(10, 5), nrows=2, ncols=1)
        ax_aggr, ax_indi = axs
        
        # ax_aggr: Aggregate distribution
        ax_aggr.hist(agg_bins[:-1], agg_bins, weights=counts_agg, color=self.colours["capital"], alpha=1, density=True)
        ax_aggr.set(ylabel=f"Prob. Density", yscale="log", ylim=ylim)
        ax_aggr.grid()
        # Ticks
        agg_x_ticks = np.floor(np.linspace(edges_agg[0], edges_agg[-1], 5))
        agg_x_ticklabels = agg_x_ticks
        self._axis_ticks_and_labels(ax_aggr, x_ticks=agg_x_ticks, x_labels=agg_x_ticklabels, y_ticks=[], y_labels=[])
        self._axis_log_ticks_and_labels(ax_aggr, exponent_range=np.log10(ylim), labels_skipped=1)
        
        # ax_indi: Individual capital distributions
        ax_indi.hist(r_indi, bins=Nbins_indi, color=self.colours["capital"],  alpha=1, density=True)
        ax_indi.set(xlabel="Return", yscale="log", ylim=ylim, ylabel="Prob. Density")
        ax_indi.grid()
        # Ticks. First set the x ticks, then the y ticks
        indi_x_ticks = [-20, -10, 0, 10, 20] # np.floor(np.linspace(edges_indi[0], edges_indi[-1], 5))
        indi_x_ticklabels = indi_x_ticks #[f"{x:.1f}" for x in indi_x_ticks]
        self._axis_ticks_and_labels(ax_indi, x_ticks=indi_x_ticks, x_labels=indi_x_ticklabels, y_ticks=[], y_labels=[])
        self._axis_log_ticks_and_labels(ax_indi, exponent_range=np.log10(ylim), labels_skipped=1)
        
        # Text, save, show
        self._text_save_show(fig, ax_aggr, "return_individual_and_aggregate", xtext=0.05, ytext=0.85, fontsize=0)
        

    def plot_multiple_return(self, group_name_list, Nbins=None, ylim=None, same_bins=True):
        """Plot the return distributions for multiple dataset (i.e. alpha=1 and N=100, alpha=4 and N=100)
        """
        # Load data
        r_arr = np.zeros(len(group_name_list), dtype=object)        
        for i, gname in enumerate(group_name_list):
            # Load data and store it in arrays
            self._get_data(gname)
            r = self._asset_return(data_name="capital_individual_mean", time_period=1)            
            r_arr[i] = r
        
        # r_arr = self._load_multiple_return_data(group_name_list)  
        
        # Bin data
        if Nbins is None:
            Nbins = int(np.sqrt(len(r_arr[0])))
        if same_bins:
            r_min = np.min([np.min(r) for r in r_arr])
            r_max = np.max([np.max(r) for r in r_arr])
            bins = np.linspace(r_min, r_max, Nbins)
        else:
            bins = Nbins
        
        # ylimits
        if ylim is None:
            ylim = (1e-4, 1)
        
        # Calculate nrows and ncols
        nrows = 2
        ncols = (len(group_name_list) + nrows - 1) // nrows
        
        # Create figure
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 5))
        
        def _plotter(index):
            # Get axis and data
            axis = axs if len(group_name_list) == 1 else axs[index//ncols, index%ncols]
            r = r_arr[index]
            # Get the parameters from the group name
            gname = group_name_list[index]
            name_dict = self._get_par_from_name(gname)
            alpha = name_dict["alpha"] 
            N = name_dict["N"]
            W = name_dict["W"]
            # Create title from parameters
            title = fr"$\alpha = {alpha}$, $N = {N}$, $W = {W}$"
            # Fit the data to a Gaussian and Student's t distribution
            x_fit, y_gauss, _, y_lst = self._return_fit(r)
            # Plot the data and the fit
            axis.plot(x_fit, y_gauss, c="k", label="Gaussian fit")
            axis.plot(x_fit, y_lst, c="grey", label="Student t fit")
            axis.hist(r, bins=bins, color=self.colours["capital"], density=True)
            axis.set_title(title, fontsize=15)
            axis.set(yscale="log", ylim=ylim)
            axis.grid()
        
            # Axis labels on outer subplots
            self._axis_labels_outer(axis, x_label="Return", y_label="Prob. Density", remove_x_ticks=same_bins)
            
        # Run the plotting function
        for i in range(len(group_name_list)):
            _plotter(i)
        
        # text save show
        if len(group_name_list) == 1:
            save_ax = axs
        else:
            save_ax = axs[0, 0]
        self._text_save_show(fig, save_ax, "multiple_return", xtext=0.05, ytext=0.85, fontsize=6)
        
        
    def KDE_and_diversity(self, bandwidth_s, bandwidth_C, C_lim, eval_points, kernel, s_lim, C_cmap="magma", percentile_cut=100, show_mean=False) -> None:
        """Plot the KDE of the wage density, the captail density and the wage diversity on three different subplots
        """
        # Get data: kde, diversity and mean diff
        s_eval, KDE_prob = self.running_KDE("salary", bandwidth_s, eval_points, kernel, s_lim)  # KDE probabilities
        C_eval, KDE_prob_C = self.running_KDE("capital", bandwidth_C, eval_points, kernel, C_lim)  # KDE probabilities
        time, diversity = self._worker_diversity()
        s, mu = self._skip_values(self.s, self.mu)
        mean_diff = mu / self.W - s.mean(axis=0)    

        # Create figure
        fig, (ax, ax_C, ax_div) = plt.subplots(figsize=(10, 10), nrows=3, gridspec_kw={'height_ratios': [2, 1, 1]})
        label_fontsize = 20
        ticks_fontsize = 12.5
        
        # Wage
        im = ax.imshow(KDE_prob, aspect="auto", origin="lower", extent=[self.skip_values, self.time_steps, np.min(s_eval), np.max(s_eval)], cmap="hot")
        if show_mean: ax.plot(time, mu/self.W, c="magenta", lw=0.6)
        ax.set_ylabel("Wage", fontsize=label_fontsize)
        ax.tick_params(axis='both', which='major', labelsize=ticks_fontsize)            
        ax.set_xticks([])
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, ticks=[], pad=-0.1, aspect=30)
        cbar.set_label(label="Frequency", fontsize=label_fontsize)
        
        # Axis ticks and ticklabels
        time_begin, time_end = self.skip_values + 50, self.time_steps - 50
        time_ticks = np.linspace(time_begin, time_end, 5)
        time_ticklabels = [time_begin, time_begin+(time_end-time_begin)/2, time_end]
        s_ticks = np.linspace(s_lim[0], s_lim[1], 5)
        s_ticklabels = [s_lim[0], s_lim[0]+(s_lim[1]-s_lim[0])/2, s_lim[1]]
        self._axis_ticks_and_labels(ax, x_ticks=[], y_ticks=s_ticks, x_labels=[], y_labels=s_ticklabels)

        # Capital
        # Plot the log probability, but because somewhere is 0, we need to add a small value to avoid log(0)
        cbar_title = "Frequency"
        
        # Normalization
        # Do not differentiate between the top 100-percentile_cut of the values
        KDE_prob_C = np.clip(KDE_prob_C, 0, np.percentile(KDE_prob_C, percentile_cut))
        
        im_C = ax_C.imshow(KDE_prob_C, aspect="auto", origin="lower", extent=[self.skip_values, self.time_steps, np.min(C_eval), np.max(C_eval)], cmap=C_cmap)
        ax_C.set_ylabel("Capital", fontsize=label_fontsize)
        ax_C.tick_params(axis='both', which='major', labelsize=ticks_fontsize)
        ax_C.set_xticks([])
        
        # Add colorbar
        cbar_C = fig.colorbar(im_C, ax=ax_C, ticks=[], pad=-0.1, aspect=15)
        cbar_C.set_label(label=cbar_title, fontsize=label_fontsize)
        # Ticks
        C_ticks = np.linspace(0, C_lim[1], 5)
        C_ticklabels = [C_ticks[0], C_ticks[0]+(C_lim[1]-C_ticks[0])/2, C_lim[1]]
        self._axis_ticks_and_labels(ax_C, x_ticks=[], y_ticks=C_ticks, x_labels=[], y_labels=C_ticklabels)

        # Diversity and mean diff
        # Diversity
        twinx = ax_div.twinx()
        twinx.plot(time, diversity, c=self.colours["diversity"])
        twinx.set(xlim=self.xlim, ylim=(0, self.N))
        twinx.set_xlabel("Time", fontsize=label_fontsize)
        twinx.set_ylabel("Wage diversity", fontsize=label_fontsize, color=self.colours["diversity"])
        twinx.tick_params(axis='y', which='major', labelsize=ticks_fontsize, labelcolor=self.colours["diversity"])
        # Ticks
        div_ticks = np.linspace(0, self.N, 5)
        div_ticklabels = [0, self.N//2, self.N]
        self._axis_ticks_and_labels(twinx, x_ticks=time_ticks, y_ticks=div_ticks, x_labels=time_ticklabels, y_labels=div_ticklabels, x_dtype="int", y_dtype="int")
        
        # Mean diff
        ax_div.plot(time, mean_diff, c=self.colours["mu"], label=r"$\mu / W - \hat{s}$", alpha=0.9)
        ax_div.grid()
        ax_div.set_ylabel(r"$\mu / W - \hat{s}$", color=self.colours["mu"], fontsize=label_fontsize)
        ax_div.set(xlim=self.xlim, ylim=(None, mean_diff.max()))
        ax_div.tick_params(axis='y', labelcolor=self.colours["mu"], labelsize=ticks_fontsize)
        # Ticks
        mean_ticks = np.round(np.linspace(0, mean_diff.max(), 5), 3)
        mean_ticklabels = mean_ticks[::2]
        self._axis_ticks_and_labels(ax_div, x_ticks=time_ticks, y_ticks=mean_ticks, x_labels=time_ticklabels, y_labels=mean_ticklabels, x_dtype="int")

        # Text, save, show
        self._text_save_show(fig, ax, "KDE_and_diversity", xtext=0.05, ytext=0.95, fontsize=0)
        
        
        
    def KDE_diversity_multiple_and_time_scale(self, group_name_list, time_scale_group_name_tensor, bandwidth_s, eval_points, kernel, s_lim) -> None:
        """The big plot that has KDE and diversity for multiple datasets, and below that ds vs frequency.

        Args:
            group_name_list (_type_): _description_
            time_scale_group_name_tensor (_type_): _description_
            bandwidth_s (_type_): _description_
            eval_points (_type_): _description_
            kernel (_type_): _description_
            s_lim (_type_): _description_
        """
        # Get the KDE and Diversity data
        self._get_data(group_name_list[0])
        KDE_arr = np.zeros((len(group_name_list), eval_points, self.time_steps-self.skip_values))
        diversity_arr = np.zeros((len(group_name_list), self.time_steps-self.skip_values))
        salary_means_diff_arr = np.zeros((len(group_name_list), self.time_steps-self.skip_values))  
        for i, gname in enumerate(group_name_list):
            # Load data and store it in arrays
            s_eval, KDE_prob = self.running_KDE("salary", bandwidth_s, eval_points, kernel, s_lim, gname=gname)
            time, diversity = self._worker_diversity(gname)
            s, mu = self._skip_values(self.s, self.mu)
            salary_means_diff = mu / self.W - np.mean(s, axis=0)
            KDE_arr[i, :, :] = KDE_prob
            diversity_arr[i, :] = diversity
            salary_means_diff_arr[i, :] = salary_means_diff
        
        fig = plt.figure(figsize=(10, 10))

        # Define height ratios explicitly: [2,1,2,1,1]
        gs = GridSpec(nrows=5, ncols=2, figure=fig, height_ratios=[2, 1, 2, 1, 1])

        # First column (top pair: 2:1 ratio)
        ax00 = fig.add_subplot(gs[0, 0])
        ax10 = fig.add_subplot(gs[1, 0])
        # Second column (top pair: 2:1 ratio)
        ax01 = fig.add_subplot(gs[0, 1])
        ax11 = fig.add_subplot(gs[1, 1])
        # First column (second pair: 2:1 ratio)
        ax20 = fig.add_subplot(gs[2, 0])
        ax30 = fig.add_subplot(gs[3, 0])
        # Second column (second pair: 2:1 ratio)
        ax21 = fig.add_subplot(gs[2, 1])
        ax31 = fig.add_subplot(gs[3, 1])
        # Row 5 spanning both columns
        ax4 = fig.add_subplot(gs[4, :])

        # Ticks
        tick_width, tick_width_minor = 1.8, 1
        tick_length, tick_length_minor = 6, 3
        
        
        time_first, time_last = self.time_values[self.skip_values] + 50, self.time_values[-1] - 50 + 1
        time_ticks = np.linspace(time_first, time_last, 5)
        time_labels = [time_first, time_first+(time_last-time_first)/2, time_last]
        
        KDE_min, KDE_max = 0, 0.18
        KDE_y_ticks = np.linspace(KDE_min, KDE_max, 5)
        KDE_y_labels= [KDE_min, KDE_min+(KDE_max-KDE_min)/2, KDE_max]
        
        div_y_ticks = np.linspace(0, self.N, 5)
        div_y_labels = [0, self.N//2, self.N]
        
        means_min, means_max = 0, 0.15
        div_means_ticks = np.linspace(means_min, means_max, 5)
        div_means_labels = [means_min, (means_max-means_min)/2, means_max]

        list_of_KDE_axis = [ax00, ax01, ax20, ax21]
        list_of_diversity_axis = [ax10, ax11, ax30, ax31]

        # Plot the KDE and diversity
        for i in range(len(group_name_list)):
            # Unpack axes and data
            ax_KDE = list_of_KDE_axis[i]
            ax_div = list_of_diversity_axis[i]
            KDE = KDE_arr[i, :, :]    
            diversity = diversity_arr[i, :]
            s_diff = salary_means_diff_arr[i, :]
            
            # For all the plotting, do the following steps:
            # 1. Plot the data
            # 2. Do the usual axis setup, e.g. limits and grid
            # 3. Add custom ticks and labels
            # 4. Remove the tick labels unless is the correct row/column
            # 5. Add the labels only to the correct outer row/column
            
            # KDE
            im = ax_KDE.imshow(KDE, aspect="auto", origin="lower", extent=[self.skip_values, self.time_steps, np.min(s_eval), np.max(s_eval)], cmap="hot")  # Plot data
            self._axis_ticks_and_labels(ax_KDE, x_ticks=time_ticks, y_ticks=KDE_y_ticks, x_labels=time_labels, y_labels=KDE_y_labels, x_dtype="int")  # Ticks and labels
            # Add y axis labels if first column, remove y tick labels from second column
            is_first_column = i % 2 == 0
            if is_first_column:
                ax_KDE.set_ylabel("Wage")
            else:
                ax_KDE.set_yticklabels([])
            # Remove all x ticks labels
            ax_KDE.set_xticklabels([])
            ax_KDE.tick_params(axis="both", width=tick_width, length=tick_length, which="major")
            ax_KDE.tick_params(axis="both", width=tick_width_minor, length=tick_length_minor, which="minor")
            
            # twin x tick label color
            # cbar pos
            # xticks on first row
            
            
            # Diversity and mean salary difference
            # Mean salary difference
            ax_div.plot(time, s_diff, c=self.colours["mu"], label=r"$\mu / W - \hat{s}$", alpha=0.9)
            ax_div.set(xlim=self.xlim, ylim=(means_min, means_max))
            ax_div.grid(which='both')
            self._axis_ticks_and_labels(ax_div, x_ticks=time_ticks, y_ticks=div_means_ticks, x_labels=time_labels, y_labels=div_means_labels, x_dtype="int")
            # Add colour to the y tick labels
            ax_div.tick_params(axis='y', labelcolor=self.colours["mu"], colors=self.colours["mu"], which="both")
            ax_div.tick_params(axis="both", width=tick_width, length=tick_length, which="major")
            ax_div.tick_params(axis="both", width=tick_width_minor, length=tick_length_minor, which="minor")
            
            # Add y axis labels, remove y tick labels form second column
            if is_first_column:
                ax_div.set_ylabel(r"$\mu / W - \hat{s}$", color=self.colours["mu"])
            else:
                ax_div.set_yticklabels([])

            # Diversity on twin axis
            twinx = ax_div.twinx()
            twinx.plot(time, diversity, label="Diversity", c=self.colours["diversity"], alpha=0.7)
            twinx.set(ylim=(0, self.N))  
            # Ticks
            self._axis_ticks_and_labels(twinx, x_ticks=time_ticks, y_ticks=div_y_ticks, x_labels=time_labels, y_labels=div_y_labels, x_dtype="int", y_dtype="int")
            twinx.tick_params(axis='y', labelcolor=self.colours["diversity"], colors=self.colours["diversity"], which="both")
            twinx.tick_params(axis="both", width=tick_width, length=tick_length, which="major")
            twinx.tick_params(axis="both", width=tick_width_minor, length=tick_length_minor, which="minor")
            
            
            # Add y axis labels if second column, remove y tick labels from first row
            if not is_first_column:
                twinx.set_ylabel("Wage diversity", color=self.colours["diversity"])
            else:
                twinx.set_yticklabels([])

            # Remove x tick labels from the first row. Add time labels to the second row
            is_first_row = i < 2
            if is_first_row:
                ax_div.set_xticklabels([])
                twinx.set_xticklabels([])
            else:
                ax_div.set_xlabel("Time")  

            
        # Plot the time scale
        marker_list = ["x", "+", "*", ".", "h", "d"]  # general_functions.list_of_markers
        ls_list = ["-", "--", "-.", ":"]
        color_list = general_functions.list_of_colors
        number_of_markers = len(marker_list)
        
        for i, group_name_arr in enumerate(time_scale_group_name_tensor):
            mean_freq1, std_freq1, mean_freq2, std_freq2, var_list = self.get_PSD_freq_multiple_var(group_name_arr, "ds", "mu")
            
            # Get the alpha, N and W values from the group name
            par_dict = self._get_par_from_name(group_name_arr[0, 0])
            alpha, N, W = par_dict["alpha"], par_dict["N"], par_dict["W"]

            # Plot data
            ax4.errorbar(var_list, mean_freq1, yerr=std_freq1, c=color_list[i % number_of_markers], linestyle=ls_list[i%number_of_markers], marker=marker_list[-(i % number_of_markers)], label=fr"$\alpha=${alpha}, $N=${N}, $W=${W}", markersize=8, alpha=0.9)
            # ax4.plot(var_list, mean_freq1, c=color_list[i % number_of_markers], linestyle=ls_list[i%number_of_markers], marker=marker_list[-(i % number_of_markers)], label=fr"$\alpha=${alpha}, $N=${N}, $W=${W}", markersize=8, alpha=0.9)
            # ax.errorbar(var_list, mean_freq2, yerr=std_freq2, c=color_list[i], fmt=marker_list[-i-1], label=fr"Second frequency, $\alpha=${alpha}, $N=${N}, $W=${W}")
        
        # Axis setup
        # Ticks
        ax4_ticks_min, ax4_ticks_max = 0, 0.06 #mean_freq1.max()*1.1
        ax4_xticks = np.linspace(0, var_list[-1], 5)
        ax4.tick_params(axis="both", width=tick_width, length=tick_length, which="major")
        ax4.tick_params(axis="both", width=tick_width_minor, length=tick_length_minor, which="minor")
        ax4.set(xlabel=r"$\Delta s / s$", ylabel="Frequency", yscale="log") #ylim=(ax4_ticks_min, ax4_ticks_max+0.0001))
        # self._axis_ticks_and_labels(ax4, x_ticks=ax4_xticks, y_ticks=np.linspace(ax4_ticks_min, ax4_ticks_max, 3), 
        #                             x_labels=ax4_xticks[::2], y_labels=[ax4_ticks_min, ax4_ticks_max])
        ax4.grid(which='major')
        self._add_legend(ax4, ncols=4, fontsize=8, x=0.5, y=1)
        
        
        # Colorbar
        # Add a new axis to the figure for the colorbar
        cbar_ax = fig.add_axes([0.095, 1, 0.82, 0.02])  # figure space [left, bottom, width, height]
        # Create a horizontal colorbar using the dedicated axis
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', ticks=[])
        cbar.ax.xaxis.set_label_position('top')  # move the label to the top
        cbar.set_label(label="Frequency", fontsize=15)
        
        # Text save show
        self._text_save_show(fig, ax00, "KDE_diversity_multiple_and_time_scale", xtext=0.05, ytext=0.95, fontsize=0)
        
        
    def plot_worker_distribution(self, Nbins=None, xlim=None, ylim=(1e-5, 2e0), xscale="log"):
        """Histogram of the counts of companies with workers.
        """
        # Get data and skip values
        self._get_data(self.group_name)
        w = self._skip_values(self.w)
        
        if xscale == "log":
            w[w==0] = 1e-1

        # Bin it
        if Nbins is None:
            Nbins = int(np.sqrt(w.size))

        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 5))
        
        if xscale == "linear":
            bins = np.linspace(w.min(), w.max(), Nbins)
            counts, _ = np.histogram(w, bins=bins, density=True)

            ax.hist(bins[:-1], bins, weights=counts, color=self.colours["workers"], alpha=1, density=True)

            x_ticks = np.linspace(bins[0], bins[-1], 5)
            x_ticklabels = x_ticks[::2]
            self._axis_ticks_and_labels(ax, x_ticks=x_ticks, y_ticks=[], x_labels=x_ticklabels, y_labels=[], x_dtype="int")

        elif xscale == "log":        
            # Binning
            bins = 10 ** np.linspace(np.log10(1e-1), np.log10(np.max(w) * 10), Nbins)  # Log x cuts off large values if max range value is not increased
            counts, edges = np.histogram(w, bins=bins, density=True)
            ax.stairs(counts, edges, color=self.colours["workers"], alpha=1)
            
            # ax.hist(bins[:-1], bins, weights=counts, color=self.colours["workers"], alpha=1, density=True)

            ax.set(xscale="log")
            # Change the xticks to match the new bin edges
            # xticks_vals = np.linspace(bins[0], bins[-1], len(bins))
            # ax.set_xticks(xticks_vals, labels=xticks_vals, fontsize=6)
            
            # x_exponent_range = (-1, np.log10(counts.max()))
            # self._axis_log_ticks_and_labels(ax, x_exponent_range, labels_skipped=1, which="x")
            
        # Setup regardless of x-scale
        ax.grid()
        ax.set(xlabel="Number of workers", ylabel="Prob. Density", yscale="log", xlim=xlim, ylim=ylim)
                        
        y_exponent_range = np.log10(ylim)
        self._axis_log_ticks_and_labels(ax, y_exponent_range, labels_skipped=1, which="y")
        
        # Text, save, show
        self._text_save_show(fig, ax, "worker_distribution", xtext=0.05, ytext=0.85, fontsize=0)

    
    def plot_worker_KDE(self, bandwidth, N_eval_points, x_eval=None, log_xscale=False, xlim=None, ylim=None):
        """KDE of worker probability density (as opposed to histogram of workers)

        Args:
            bandwidth (_type_): _description_
            N_eval_points (_type_): _description_
            x_eval (_type_, optional): _description_. Defaults to None.
            log_xscale (bool, optional): _description_. Defaults to False.
            xlim (_type_, optional): _description_. Defaults to None.
            ylim (_type_, optional): _description_. Defaults to None.
        """
        
        # Get data
        self._get_data(self.group_name)
        
        # Skip values and prepare for log
        w = self._skip_values(self.w)
        w = np.where(w == 0, 1e-1, w)
        w = w.ravel()
        
        # KDE
        x_eval, P = self.onedim_KDE(w, bandwidth, x_eval=x_eval, N_eval_points=N_eval_points, kernel="epanechnikov", log_scale=log_xscale)
        
        fig, ax = plt.subplots(figsize=(10, 5)) 
        ax.plot(x_eval, P, c=self.colours["workers"])
        ax.set(xlabel="Number of workers", ylabel="Prob. Density", yscale="log", xlim=xlim, ylim=ylim)
        if log_xscale:
            ax.set(xscale="log")
        ax.grid()
        
        # Text save show
        self._text_save_show(fig, ax, "worker_KDE", xtext=0.05, ytext=0.85, fontsize=0)
        
        
    def plot_increased_decreased(self, window_size=5, bandwidth_s=0.004, s_lim=(0.000, 0.18), eval_points=400, kernel="epanechnikov"):
        """Plot the number of companies who increased vs decreased their wages together with the wage density
        """
        # Get data
        self._get_data(self.group_name)
        # KDE
        s_eval, KDE_prob = self.running_KDE("salary", bandwidth_s, eval_points, kernel, s_lim)  # KDE probabilities

        # Calculate capital diff
        C, time = self._skip_values(-self.d, self.time_values[:-1])
        C_diff = np.diff(C)
        increased = np.count_nonzero(C_diff>0, axis=0)
        increased_at_zero = np.count_nonzero(C_diff==0, axis=0)
        decreased = np.count_nonzero(C_diff<0, axis=0)
        # Calculate rolling averages
        increased = uniform_filter1d(increased, size=window_size)
        increased_at_zero = uniform_filter1d(increased_at_zero, size=window_size)
        decreased = uniform_filter1d(decreased, size=window_size)
        
        # Create figure
        fig, (ax_w, ax) = plt.subplots(figsize=(10, 5), nrows=2)
        # KDE
        im = ax_w.imshow(KDE_prob, aspect="auto", origin="lower", extent=[self.skip_values, self.time_steps, np.min(s_eval), np.max(s_eval)], cmap="hot")


        ax.plot(time, increased, "-", color="green", label=r"Increased $w$")
        ax.plot(time, increased_at_zero, ls="dashdot", color="orange", label=r"Increased $w$ at $\Delta C =0$")
        ax.plot(time, decreased, "--", color="red", label=r"Decreased $w$")
        ax.set(xlabel="Time", ylabel="Number of companies", xlim=self.xlim)
        ax.grid()
        
        
        
        
        self._add_legend(ax, ncols=3, fontsize=8, x=0.5, y=1)
        
        # Text save show
        self._text_save_show(fig, ax, "increased_decreased", xtext=0.05, ytext=0.85, fontsize=8)
        