import general_functions 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import functools
import numpy as np
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
        fig, ax0 = plt.subplots(nrows=nrows, figsize=(10, 10))
        
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
        ax0.set(xlim=self.xlim, ylabel="Log Price", yscale="linear", title="Mean salary and bankruptcies")
        ax0.set_ylabel("Log Price", color=c0)
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
        self.d = -self.d
        mean_debt = self.d.mean(axis=0)[self.skip_values:]
        median_debt = np.median(self.d, axis=0)[self.skip_values:]
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
        fig, ax = plt.subplots(figsize=(10, 6))
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

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.errorbar(m_vals, mean_min_arr, yerr=std_mean_min_arr, fmt="v", label=r"$\min(\bar{s}) / m $", color="k")
        ax.errorbar(m_vals, mean_max_arr, yerr=std_mean_max_arr, fmt="^", label=r"$\max(\bar{s}) / m $", color="k")
        ax.set(xlabel=r"$m$", ylabel="Price", yscale="log", xscale="log")
        # ax.set_title(f"Repeated measurements of min and max salary, N={N_repeats}, t={time_steps}", fontsize=10)
        ax.grid()
        
        self._add_legend(ax, ncols=2, y=0.9)
        
        # Text, save show
        save_name = "min_max_salary_vs_m"
        # Include the last minimum salary taken from the group name to the save name
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
            fig, ax = plt.subplots(figsize=(12, 8))
            for i, (mean_salary, expo) in enumerate(zip(mean_salary_list, prob_expo_list)):
                ax.plot(time_vals, mean_salary, label=f"Exponent = {int(expo)}", c=c_list[i])
            ax.set(title="Mean salary for different probability exponents", xlabel="Time", ylabel="Price", yscale="linear")
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
                ax.plot(time_vals, mean_salary, c=c_list[i])
                ax.set_title(fr"Exponent = {int(expo)}", fontsize=8)
                ax.set(yscale="linear")
                ax.grid()

            # Axis labels. Only the bottom row should have x labels, and only the left column should have y labels
            subplot_spec = ax.get_subplotspec()
            if subplot_spec.is_last_row():
                ax.set_xlabel("Time")
            if subplot_spec.is_first_col():
                ax.set_ylabel("Log Price")
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
    
    
    def _load_multiple_variable_repeated(self, group_name_arr, data_name: str):
        """Loop over the 2d array group_name_arr and store the mean salary for each group name in an array.

        Args:
            group_name_arr (np.ndarray): 2d array with group names. Rows are variable values, columns are repeated runs.
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


    def plot_var_frequency(self, group_name_list: list, var_name: str, data_name: str):
        """Use the PSD to find the frequency of the oscillation in the data set for different var_name values.

        Args:
            group_name_list (list): _description_
            var_name (str): Either "ds" or "alpha". Determines what variable to plot against.
            data_name (str): Either "salary" or "mu". Determines what data to load and find the frequency on.
        """
        assert var_name in ["ds", "alpha"], f"var_name must be either 'ds' or 'alpha', not {var_name}"
        assert data_name in ["salary", "mu"], f"data_name must be either 'salary' or 'mu', not {data_name}"
        
        # Load data
        self._get_data(group_name_list[0])
        time_vals = np.arange(self.skip_values, self.time_steps)
        data_list = np.zeros((len(group_name_list), len(time_vals)))
        var_list = np.zeros(len(group_name_list))
        
        for i, gname in enumerate(group_name_list):
            # Get values
            self._get_data(gname)
            
            if data_name == "salary":
                data = np.mean(self.s[:, self.skip_values:], axis=0)
            elif data_name == "mu":
                data = self.mu[self.skip_values:]
            if var_name == "ds":
                var_value = self.ds
            elif var_name == "alpha":
                var_value = self.prob_expo
            
            # Append values
            data_list[i] = data
            var_list[i] = var_value
        
        # For each data set, using PSD find the frequency of the oscillation by taking the max frequency of the two most dominant frequencies
        freq_list = np.zeros(len(var_list))
        freq2_list = np.zeros(len(var_list))
        for i, mean_salary in enumerate(data_list):
            freq, psd = self._PSD_on_dataset(mean_salary, number_of_frequencies=2, fs=1)
            # freq_list has the most prominent frequency, and freq2_list has the second most prominent frequency
            freq_list[i] = freq[0]
            freq2_list[i] = freq[1]
        
        # Linear fit to dominant frequency data        
        par, cov = np.polyfit(var_list, freq_list, deg=1, cov=True)
        std = np.sqrt(np.diag(cov))
        x_fit = np.linspace(np.min(var_list), np.max(var_list), 100)
        y_fit = par[0] * x_fit + par[1]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(var_list, freq_list, "o", label="Most prominent frequency")
        ax.plot(var_list, freq2_list, "x", label="Second most prominent frequency")
        ax.plot(x_fit, y_fit, ls="--", label=r"Linear fit")
        ax.set(xlabel=f"{var_name}", ylabel="Frequency", title=fr"{data_name} oscillation frequency")
        self._add_legend(ax, ncols=3, x=0.5, y=0.95)
        ax.grid()
        
        # Print the fit parameters with their uncertainty
        fit_text = fr"$a = $ {par[0]:.3f} $\pm$ {std[0]:.3f}, $b = $ {par[1]:.2f} $\pm$ {std[1]:.5f}"
        ax.text(0.95, 0.85, fit_text, transform=ax.transAxes, fontsize=8, horizontalalignment="right")
        print(fit_text)
        
        # Text, save show,
        self._text_save_show(fig, ax, f"frequency_{var_name}_{data_name}", xtext=0.05, ytext=0.75, fontsize=7)


    def plot_ds_power_spectrum(self, group_name_list):
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
        
        for i, (psd, freq, ds) in enumerate(zip(psd_list, freq_list, ds_list)):
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
        
        fig.suptitle("Power Spectral Density of mean salary for different ds values")
        # Text, save show,
        self._text_save_show(fig, axs[0, 0], "ds_power_spectrum", xtext=0.05, ytext=0.75, fontsize=6)


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
        

    def plot_N_W_ratio(self, group_name_list):
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
        nrows = 2
        ncols = (len(group_name_list) + nrows - 1) // nrows
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 8))
        
        for i, (mean_salary, bankruptcy, N, W) in enumerate(zip(mean_salary_list, bankruptcy_list, N_list, W_list)):
            ax = axs[i//ncols, i%ncols]
            twin_x = ax.twinx()
            twin_x.plot(time_vals, bankruptcy, c=self.colours["bankruptcy"], alpha=0.5)
            twin_x.set_ylabel("Fraction bankrupt", color=self.colours["bankruptcy"])
            twin_x.tick_params(axis='y', labelcolor=self.colours["bankruptcy"])
            
            ax.plot(time_vals, mean_salary, c=self.colours["salary"])
            ax.set_title(fr"$W/N = {W:.0f}/{N:.0f}$", fontsize=8)
            ax.set(yscale="log")
            ax.grid()

            # Axis labels. Only the bottom row should have x labels, and only the left column should have y labels
            subplot_spec = ax.get_subplotspec()
            if subplot_spec.is_last_row():
                ax.set_xlabel("Time")
            if subplot_spec.is_first_col():
                ax.set_ylabel("Log Price")
        
        fig.suptitle(fr"Mean salary for $W/N =$ {N_W_ratio}")
        
        # Text, save show,
        self._text_save_show(fig, axs[0, 0], "N_W_ratio", xtext=0.05, ytext=0.75, fontsize=6)
        
        
    def plot_N_var_W_const(self, group_name_list):
        """Plot the mean salary for N values, each on their own subplot
        """       
        # Load data
        self._get_data(group_name_list[0])
        time_vals = np.arange(self.skip_values, self.time_steps)
        mean_salary_list = np.zeros((len(group_name_list), len(time_vals)))
        N_list = np.zeros(len(group_name_list))
        for i, gname in enumerate(group_name_list):
            # Get values
            self._get_data(gname)
            mean_salary = np.mean(self.s[:, self.skip_values:], axis=0)
            N = self.N
            # Append values
            mean_salary_list[i] = mean_salary
            N_list[i] = N
        # Create figure
        # Calculate nrows and ncols
        nrows = 2
        ncols = (len(group_name_list) + nrows - 1) // nrows
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 8))
        
        for i, (mean_salary, N) in enumerate(zip(mean_salary_list, N_list)):
            ax = axs[i//ncols, i%ncols]
            ax.plot(time_vals, mean_salary, c=self.colours["salary"])
            ax.set_title(fr"$W/N = {self.W:.0f}/{N:.0f}$", fontsize=8)
            ax.set(yscale="log")
            ax.grid()

            # Axis labels. Only the bottom row should have x labels, and only the left column should have y labels
            subplot_spec = ax.get_subplotspec()
            if subplot_spec.is_last_row():
                ax.set_xlabel("Time")
            if subplot_spec.is_first_col():
                ax.set_ylabel("Log Price")
        
        fig.suptitle(fr"Mean salary for $N$ variable, $W = {self.W}$")
        
        # Text, save show,
        self._text_save_show(fig, axs[0, 0], "N_var_W_const", xtext=0.05, ytext=0.75, fontsize=6)


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
            ax.set(yscale="log")
            ax.grid()

            # Axis labels. Only the bottom row should have x labels, and only the left column should have y labels
            subplot_spec = ax.get_subplotspec()
            if subplot_spec.is_last_row():
                ax.set_xlabel("Time")
            if subplot_spec.is_first_col():
                ax.set_ylabel("Log Price")
        
        fig.suptitle(fr"Mean salary for $W$ variable, $N = {self.N}$")
        
        # Text, save show,
        self._text_save_show(fig, axs[0, 0], "N_const_W_var", xtext=0.05, ytext=0.75, fontsize=6)

        
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
        d_eval, KDE_prob_d = self.running_KDE("debt", bandwidth_d, eval_points, kernel, d_lim)  # KDE probabilities
        time_values, mu, d_mean = self._skip_values(self.time_values, self.mu, self.d.mean(axis=0))  # Get mean salary and skip values
        
        # Create figure
        figsize = (16, 12) if plot_debt else (10, 8)
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
        if show_mean: ax.plot(time_values, mu/self.W, c="magenta", label=r"$\mu / W$", alpha=1, lw=0.6)
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
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(time_vals, diversity, c=self.colours["diversity"])
        ax.set(xlabel="Time", ylabel="Diversity")
        ax.grid()
        
        # Text, save, show
        self._text_save_show(fig, ax, "diversity", xtext=0.05, ytext=0.85, fontsize=6)
        
        
    def plot_diversity_multiple_alpha(self, group_name_list):
        """Plot the worker diversity for different alpha values, each on their own subplot
        """       
        # Load data
        self._get_data(group_name_list[0])
        time_vals = np.arange(self.skip_values, self.time_steps)
        diversity_list = []
        alpha_list = []
        for gname in group_name_list:
            # Get values
            self._get_data(gname)
            print(f"Calculating diversity for {gname}")
            print(self.prob_expo)
            # time_vals, diversity = self._worker_diversity()
            
            # Append values
            # diversity_list.append(diversity)
            diversity_list.append(np.mean(self.s[:, self.skip_values:], axis=0))
            alpha_list.append(self.prob_expo)
        
        # Create figure
        ncols = 2
        nrows = (len(group_name_list) + ncols - 1) // ncols
        fig, axs = plt.subplots(figsize=(12, 8), ncols=ncols, nrows=nrows)
        
        # Loop over axes
        for i in range(len(group_name_list)):
            ax = axs[i//ncols, i%ncols]
            ax.plot(time_vals, diversity_list[i], c=self.colours["diversity"])
            ax.set_title(fr"$\alpha = {alpha_list[i]:.0f}$", fontsize=8)
            ax.grid()
            
            # Axis labels
            subplot_spec = ax.get_subplotspec()
            if subplot_spec.is_last_row():
                ax.set_xlabel("Time")
            if subplot_spec.is_first_col():
                ax.set_ylabel("Diversity")
        
        # Text save show
        self._text_save_show(fig, axs[0, 0], "diversity_multiple_alpha", xtext=0.05, ytext=0.85, fontsize=6)