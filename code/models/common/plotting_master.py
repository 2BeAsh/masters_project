# Plotting
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib.colors import PowerNorm
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, NullFormatter, LogFormatterMathtext, LogLocator, SymmetricalLogLocator, LogFormatter, FixedLocator
# Numerical
import numpy as np
import pandas as pd
import scipy.optimize
from scipy.stats import norm
from scipy.ndimage import uniform_filter1d
import scipy.stats
# My scripts
import general_functions 
from run import dir_path_image
from postprocess import PostProcess
# Other
import functools
from tabulate import tabulate
from tqdm import tqdm


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
            "bankruptcy": "red",
            "salary": general_functions.list_of_colors[0],
            "debt": general_functions.list_of_colors[1],
            "interest_rate": general_functions.list_of_colors[2],
            "workers": general_functions.list_of_colors[3],
            "mutations": general_functions.list_of_colors[4],
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


    def plot_salary(self, show_profit=False, window_size=1, axis=None, xlim=None, yscale="linear"):
        """Plot the mean salary and fraction who went bankrupt on twinx. Plot the spread (std/mean) on a subplot below it."""
        self._get_data(self.group_name)
        # mean_salary = self.s.mean(axis=0)[self.skip_values:]
        
        mu, s, w, went_bankrupt, time_values = self._skip_values(self.mu, self.s, self.w, self.went_bankrupt, self.time_values)
        mu = mu / (self.W)
        # median_salary = np.median(self.s, axis=0)[self.skip_values:]
        N_nonw0 = self.N - np.count_nonzero(w==0, axis=0)
        fraction_bankrupt = (went_bankrupt / N_nonw0)
        
        above_1_idx = np.where(fraction_bankrupt>1)
        # spread = (self.s.std(axis=0)[self.skip_values:] / mean_salary)
        
        # Rolling averages
        fraction_bankrupt = uniform_filter1d(fraction_bankrupt, size=window_size)
        mu = uniform_filter1d(mu, size=window_size)
        
        # Create figure
        if axis is None:
            nrows = 1 if not show_profit else 2
            fig, ax0 = plt.subplots(nrows=nrows, figsize=(10, 5))
        
        if show_profit:
            ax0, ax1 = ax0
        
        # ax0 - Salary and fraction who went bankrupt
        c0 = self.colours["mu"]
        c1 = self.colours["bankruptcy"]
        
        # Bankruptcy
        ax0_twin = ax0.twinx()
        ax0_twin.plot(time_values, fraction_bankrupt, color=c1, label="Fraction bankrupt", alpha=0.5)
        ax0_twin.set_ylabel("Fraction bankrupt", color=c1)
        ax0_twin.tick_params(axis='y', labelcolor=c1)

        # Mean and median salary
        # ax0.plot(time_values, mean_salary, label="Mean salary", c=c0, alpha=1)
        ax0.plot(time_values, mu, label=r"$P/W$", c=self.colours["mu"])
        # ax0.plot(time_values, median_salary, label="Median salary", c="black", alpha=0.7, ls="dotted")
        if xlim is None: xlim = self.xlim
        ax0.set(xlim=xlim, xlabel="Time [a.u.]", yscale=yscale, )#title="Mean salary and bankruptcies")
        if yscale == "linear": ylabel = r"$P / W$" 
        else: ylabel = r"Log $P / W$"
        ax0.set_ylabel(ylabel, color=c0)
        ax0.tick_params(axis='y', labelcolor=c0)
        ax0.grid()
        # self._add_legend(ax0, ncols=3, x=0.5, y=0.9)
        
        if show_profit:
            # ax1 - profit
            mean_s = np.mean(s, axis=0)  # Average over company
            ax1.plot(time_values, mu - mean_s, c=self.colours["mu"])
            ax1.set(xlabel="Time [a.u.]", xlim=self.xlim, ylabel=r"$P / W - \bar{s}$")
            ax0.set(xlabel="")
            ax1.grid()
            if yscale == "log":
                ax1.set_yscale("symlog", linthresh=1e-2)
                ax1.yaxis.set_major_locator(FixedLocator([-1e10, -1e5, 0, 1e5, 1e10,]))
                ax1.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
        
            # Plot the peaks as vertical lines on ax0 and ax1
            if np.any(self.peak_idx != None):
                for peak in self.peak_idx:
                    ax0.axvline(x=peak, ls="--", c="grey", alpha=0.7)
                    ax1.axvline(x=peak, ls="--", c="grey", alpha=0.7)
            self._subplot_label(ax1, 1)

        self._subplot_label(ax0, 0)
        
        self._text_save_show(fig, ax0, "salary", xtext=0.05, ytext=0.95, fontsize=6)
        
        
    def plot_single_companies(self, N_plot):
        """Plot the salary and debt of the first N_plot companies"""
        # Get data and remove skip_values
        self._get_data(self.group_name)
        
        # Get first N_plot companies
        s = self.s[:N_plot, self.skip_values:]
        C = -self.d[:N_plot, self.skip_values:]
        w = self.w[:N_plot, self.skip_values:]
        time_values = np.arange(self.skip_values, self.time_steps)
                
        # Create figure
        fig, (ax_s, ax_d, ax_w) = plt.subplots(nrows=3, figsize=(10, 5))
        
        # ax_s - salary
        ylim = (0.99e-2, np.max(s)*1.01)
        ax_s.plot(time_values, s.T, color=self.colours["salary"])
        # ax_s.set(title=f"Salary and debt of first {N_plot} companies", yscale="linear", ylim=ylim)
        ax_s.set(yscale="linear", ylim=ylim, ylabel="Wage [$]")
        ax_s.grid()
        
        # ax_d - debt
        ax_d.plot(time_values, C.T, c=self.colours["capital"])
        ax_d.set(ylabel="Capital [$]", yscale="linear")
        ax_d.grid()

        # Plot bankruptcies for the first company on the debt subplot
        idx_bankrupt = self.went_bankrupt_idx[0, self.skip_values:]
        time_bankrupt = time_values[idx_bankrupt]
        d_bankrupt = C[0, idx_bankrupt]
        s_bankrupt = s[0, idx_bankrupt]
        ax_d.scatter(time_bankrupt, d_bankrupt, c=self.colours["bankruptcy"], marker="x", s=20)
        ax_s.scatter(time_bankrupt, s_bankrupt, c=self.colours["bankruptcy"], marker="x", s=20)

        # ax_w - workers
        ax_w.plot(time_values, w.T, c=self.colours["workers"])
        ax_w.set(xlabel="Time", ylabel="Workers", yscale="linear")
        ax_w.grid()
        
        # Ticks
        x_ticks = [2100, 2200, 2300, 2400, 2500]
        x_labels = x_ticks[::2]
        for i, axis in enumerate((ax_s, ax_d, ax_w)):
            self._axis_ticks_and_labels(axis, x_ticks=x_ticks, x_labels=x_labels, x_dtype="int")
            self._subplot_label(axis, i)
            
        
        # Subplot labels
        
        self._text_save_show(fig, ax_s, "single_companies", xtext=0.05, ytext=0.85)
        
        
    def plot_capital(self):
        """Plot the mean debt and fraction who went bankrupt on twinx and below it debt together with salary, last subplot has debt distribution at final time step. 
        """
        # Preprocess
        self._get_data(self.group_name)
        C = -self.d
        mean_C = C.mean(axis=0)[self.skip_values:]
        median_C = np.median(C, axis=0)[self.skip_values:]
        # mean_salary = self.s.mean(axis=0)[self.skip_values:]
        mu = self._skip_values(self.mu,)
        mean_salary = mu / (self.W )
        fraction_bankrupt = (self.went_bankrupt[self.skip_values:] / self.N)
        time_values = np.arange(self.skip_values, self.time_steps)
        C_final = C[:, -1]
        
        # Create figure
        fig, (ax, ax1) = plt.subplots(nrows=2, figsize=(10, 5))
        c0 = self.colours["capital"]
        c1 = self.colours["bankruptcy"]
        
        ax.plot(time_values, mean_C, c=c0, label="Mean Capital")
        # ax.plot(time_values, median_debt, c=c0, ls="--", label="Median debt")
        ax.set(xlabel="Time", title="Mean capital and bankruptcies", yscale="linear", xlim=self.xlim)
        ax.set_ylabel("Price", color=c0)
        ax.tick_params(axis='y', labelcolor=c0)
        ax.grid()
        
        ax_twin = ax.twinx()
        ax_twin.plot(time_values, fraction_bankrupt, color=c1, label="Fraction bankrupt", alpha=0.6)
        ax_twin.set_ylabel("Fraction bankrupt", color=c1)
        ax_twin.tick_params(axis='y', labelcolor=c1)
        
        # ax1 - Salary and debt
        c2 = self.colours["mu"]
        ax1.plot(time_values, mean_C, c=c0)
        ax1.set(xlabel="Time", title="Mean salary and capital", yscale="linear", xlim=self.xlim)
        ax1.set_ylabel("Mean Capital", color=c0)
        ax1.tick_params(axis='y', labelcolor=c0)
        ax1.grid()
        
        ax1_twin = ax1.twinx()
        ax1_twin.set(xlim=self.xlim)
        ax1_twin.plot(time_values, mean_salary, c=c2, alpha=0.7)
        ax1_twin.set_ylabel(r"$P / W$", color=c2)
        ax1_twin.tick_params(axis='y', labelcolor=c2)
        ax1_twin.set_yscale("linear")
        
        # ax2 - Debt distribution
        # Nbins = int(np.sqrt(self.N))
        # ax2.hist(d_final, bins=Nbins, color=c0)
        # ax2.set(title="Debt distribution at final time step", xlabel="Debt", ylabel="Counts", yscale="log")
        # ax2.grid()
        
        # Log scale hist requires only positive values
        # self._xlog_hist(d_final, fig, ax2, xlabel="Log Debt", ylabel="Counts", title="Debt distribution at final time step")
        
        self._text_save_show(fig, ax, "capital", xtext=0.05, ytext=0.85, fontsize=6)
    
    
    def plot_mu_mean_s_diversity(self):
        """Plot the system money spent mu over time, and the mean salary on a twinx axis.
        """
        # Get data
        self._get_data(self.group_name)
        _, diversity = self._worker_diversity()  # Already time skipped
        time_values, mu, s = self._skip_values(self.time_values, self.mu, self.s)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(time_values, mu/self.W - s.mean(axis=0), c=self.colours["mu"], label=r"$P / W - \hat{s}$", alpha=0.9)
        ax.tick_params(axis='y', labelcolor=self.colours["mu"])
        ax.set_ylabel(r"$P / W - \hat{s}$", color=self.colours["mu"])
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
    
    
    def plot_min_max_vs_m(self, group_name_arr_linear, data_name: str, smooth_size=1, time_points_to_show=500):
        """Plot the mean of repeated measurements of the minimum and maximum of the mean salary, together with their uncertanties.
        """
        # fig, ax_arr = plt.subplots(figsize=(10, 10), nrows=2, ncols=2, height_ratios=[2, 1], sharey="row", sharex="row", )
        fig, ax_arr = plt.subplots(figsize=(10, 10), nrows=2, ncols=1, height_ratios=[2, 1], sharey="row", sharex="row", )
        
        def _load_and_plot(gname_arr, axis_tuple):
            # Calculate mean and std of min and max salary for each m
            self._get_data(gname_arr[0, 0])
            mean_salary_arr, variable_dict = self._load_multiple_variable_repeated(gname_arr, data_name)
            m_vals = variable_dict["m"]
            
            N_repeats = np.shape(mean_salary_arr)[1]
            time_steps = np.shape(mean_salary_arr)[2]
            time_values = np.arange(self.skip_values, time_steps+self.skip_values)
            
            min_arr = np.min(mean_salary_arr, axis=2)
            mean_min_arr = np.mean(min_arr, axis=1) / m_vals  # Normalize the minimum salary by the mutation magnitude
            std_mean_min_arr = np.std(min_arr, axis=1, ddof=1) / np.sqrt(N_repeats) / m_vals
            
            max_arr = np.max(mean_salary_arr, axis=2) 
            mean_max_arr = np.mean(max_arr, axis=1) / m_vals  # Normalize the maximum salary by the mutation magnitude
            std_mean_max_arr = np.std(max_arr, axis=1, ddof=1) / np.sqrt(N_repeats) / m_vals
            
            # Plotting
            ax_ts, ax_extr = axis_tuple
            c_list = general_functions.list_of_colors
            # Time series
            # Take the first of the reapeated measurement graphs and plot for each m value
            for i, m in enumerate(m_vals):
                mu = mean_salary_arr[i, 0, :]  # Get data to the i'th m value for the 0'th repeated measurement
                mu_smooth = uniform_filter1d(mu, size=smooth_size)
                mu_show = mu_smooth[:time_points_to_show]
                mu_show_norm = mu_show / np.mean(mu_show)
                
                text = self.format_scientific_latex(m, precision=0, include_mantissa=False)
                text_full = fr"$m = {text}$"
                ax_ts.plot(time_values[:time_points_to_show], mu_show_norm, alpha=0.9, c=c_list[i], label=text_full)
                # Write m-value in text
                # y_text = np.mean(mu_smooth)
                # x_text = time_values[0] - 25
                # ax_ts.text(x_text, y_text, s=text_full, fontsize=10, rotation=25)

            ax_ts.set(xlabel="Time [a.u.]", ylabel=r"Normalized Average Price $P/W$", yscale="linear")
            ax_ts.grid()
            ax_ts.legend(frameon=False)
            # Extremum
            label_min = r"Minimum"
            label_max = r"Maximum"
            ax_extr.errorbar(m_vals, mean_min_arr, yerr=std_mean_min_arr, fmt="v", label=label_min, color="k")
            ax_extr.errorbar(m_vals, mean_max_arr, yerr=std_mean_max_arr, fmt="^", label=label_max, color="k")
            ax_extr.set(xlabel=r"$m$", ylabel=r"Average Price relative to $m$", yscale="linear", xscale="log")
            ax_extr.grid()
            y_ticks_extr = [0.5, 1, 1.5, 2]
            self._axis_ticks_and_labels(ax_extr, y_ticks=y_ticks_extr, y_labels=y_ticks_extr)
            
            # self._add_legend(ax_extr, ncols=2, y=0.9, fontsize=15)
            ax_extr.legend(frameon=False, loc="center right")
            
            # Subplot labels
            self._subplot_label(ax_ts, index=0)
            self._subplot_label(ax_extr, index=1, location=(0.05, 0.5))
        
        # Run the plotter function
        _load_and_plot(group_name_arr_linear, ax_arr)
        # _load_and_plot(group_name_arr, (ax_arr[0, 0], ax_arr[1, 0]))
        # _load_and_plot(group_name_arr_linear, (ax_arr[0, 1], ax_arr[1, 1]))
        
        # Text, save show
        save_name = "min_max_salary_vs_m"
        # Include the last minimum salary taken from the group name to the save name
        last_s_min = group_name_arr_linear[-1, -1].split("_")[-2]
        combined_save_name = save_name + last_s_min
        self._save_fig(fig, combined_save_name)
        plt.show()


    def plot_smin_m_ratio(self, group_name_arr, smooth_size=1, time_points_to_show=500):
        
        fig, ax_tuple = plt.subplots(figsize=(10, 10), height_ratios=[2, 1], ncols=1, nrows=2)
        
        def _load_and_plot(gname_arr, axis_tuple):
            # Calculate mean and std of min and max salary for each m
            self._get_data(gname_arr[0, 0])
            mean_salary_arr, variable_dict = self._load_multiple_variable_repeated(gname_arr, "mu")
            m_vals = variable_dict["m"]
            smin_vals = variable_dict["smin"]
            
            N_repeats = np.shape(mean_salary_arr)[1]
            time_steps = np.shape(mean_salary_arr)[2]
            time_values = np.arange(self.skip_values, time_steps+self.skip_values)
            
            min_arr = np.min(mean_salary_arr, axis=2)
            mean_min_arr = np.mean(min_arr, axis=1) / m_vals   # Normalize the minimum salary by the mutation magnitude
            std_mean_min_arr = np.std(min_arr, axis=1, ddof=1) / np.sqrt(N_repeats) / m_vals 
            
            max_arr = np.max(mean_salary_arr, axis=2) 
            mean_max_arr = np.mean(max_arr, axis=1) / m_vals   # Normalize the maximum salary by the mutation magnitude
            std_mean_max_arr = np.std(max_arr, axis=1, ddof=1) / np.sqrt(N_repeats) / m_vals 
            
            # Plotting
            ax_ts, ax_extr = axis_tuple
            
            # Time series
            # Take the first of the reapeated measurement graphs and plot for each m value
            c_list = general_functions.list_of_colors
            for i, smin in enumerate(smin_vals):
                smin_factor_text = self.format_scientific_latex(smin/m_vals[i], precision=0, include_mantissa=False)
                text_full = fr"$s_\text{{min}}/m = {smin_factor_text}$"

                # Write m-value in text
                mu = mean_salary_arr[i, 0, :]  # Get data to the i'th m value for the 0'th repeated measurement
                mu_smooth = uniform_filter1d(mu, size=smooth_size)
                mu_show = mu_smooth[:time_points_to_show]
                ax_ts.plot(time_values[:time_points_to_show], mu_show, alpha=0.8, c=c_list[i], label=text_full)
                y_text = mu_show[0]
                x_text = time_values[0] - 15
                # ax_ts.text(x_text, y_text, s=text_full, fontsize=12, rotation=25, fontweight="bold", color=c_list[i], backgroundcolor=(224/255,224/255,224/255))
            
            ax_ts.set(xlabel="Time [a.u.]", ylabel="Price [a.u.]", yscale="linear")
            ax_ts.grid()
            # Add a horizontal line to indicate m
            m_text = fr"$m = {m_vals[0]}$"
            # ax_ts.text(x_text, y=1.05*m_vals[0], s=m_text, fontsize=12, color="grey")
            ax_ts.axhline(y=m_vals[0], ls="dashed", color="grey", label=m_text)
            ax_ts.legend(frameon=False, loc="upper right")
            
            # Extremum
            label_min = r"Minimum"
            label_max = r"Maxmimum"
            ax_extr.errorbar(smin_vals/m_vals, mean_min_arr, yerr=std_mean_min_arr, fmt="v", label=label_min, color="k")
            ax_extr.errorbar(smin_vals/m_vals, mean_max_arr, yerr=std_mean_max_arr, fmt="^", label=label_max, color="k")
            ax_extr.set(xlabel=r"$s_\text{min}/m$", ylabel="Relative Price", yscale="linear", xscale="log")
            ax_extr.grid()
            yticks = [0.5, 1, 1.5, 2]
            yticklabels = yticks
            self._axis_ticks_and_labels(ax_extr, y_ticks=yticks, y_labels=yticklabels)
            
            # self._add_legend(ax_extr, ncols=2, y=0.9, fontsize=15)
            ax_extr.legend(frameon=False, loc="upper left")
            
            # Subplot labels
            self._subplot_label(ax_ts, index=0)
            self._subplot_label(ax_extr, index=1, location=(0.05, 0.5))
        
        
        _load_and_plot(group_name_arr, axis_tuple=ax_tuple)
        
        
        # Text, save show,
        self._text_save_show(fig, ax_tuple[0], f"smin_m_ratio", xtext=0.05, ytext=0.75, fontsize=1)



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
        smin_arr = np.zeros_like(alpha_arr)
        
        for i in range(group_name_arr.shape[0]):
            for j in range(group_name_arr.shape[1]):
                gname = group_name_arr[i, j]
                self._get_data(gname)
                if data_name == "salary":
                    data = np.mean(self.s[:, self.skip_values:], axis=0)
                elif data_name == "mu":
                    data = self.mu[self.skip_values:] / self.W
                data_arr[i, j] = data
            ds_arr[i] = self.ds
            m_arr[i] = self.m
            alpha_arr[i] = self.prob_expo
            smin_arr[i] = self.s_min
        
        variable_dict = {"ds": ds_arr, "m": m_arr, "alpha": alpha_arr, "smin": smin_arr}
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
            


    def plot_var_frequency(self, group_name_arr, var_name: str, data_name: str, points_to_exclude_from_fit=0, show_second_dominant_freq=False, show_fit_results=False):
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
        d = -self.d[:, -1]
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
        ax_d.set(xlabel="Capital", yscale="log", ylabel="Counts")
        ax_s.set_title("Salary distribution", fontsize=8)
        ax_d.set_title("Capital distribution", fontsize=8)
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
            ax.set_xlabel(r"Log $s_b - P$")
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
        if show_mean: ax.plot(time_values, mu/self.W, c="magenta", label=r"$P / W$", alpha=1, lw=1)
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
    
    
    def plot_effective_smin(self):
        """Look at the highest salaries of companies with no workers and the lowest salaries of companies with 1 worker to estimate the effective minimum wage.
        """
        # Get data
        self._get_data(self.group_name)
        s, w, mu, time_steps, time_values = self._skip_values(self.s, self.w, self.mu, self.time_steps, self.time_values)
        
        # For each time point, find the top 5% wages with w=0 and the lowest 5% wages with w=1 (if possible)
        s_top5_w0 = np.empty(time_steps)
        s_low5_w1 = np.empty(time_steps)
        for i in range(time_steps):
            w_i = w[:, i]
            s_i = s[:, i]
            # Calculate the top 5% wages of the w = 0 companies
            w0 = w_i[w_i == 0]
            if len(w0) == 0:
                s_top5_w0[i] = np.nan
            else:
                s_i_w0 = s_i[w0]
                s_i_w0_sort = np.sort(s_i_w0)
                n_in_top5 = int(np.max((0.95 * len(s_i_w0_sort), len(s_i_w0_sort)-1)))  # Ensure has at least 1 company
                s_i_w0_top5 = s_i_w0_sort[n_in_top5:]  # Get top 5% best
                s_i_w0_top5_mean = np.mean(s_i_w0_top5)
                # s_top5_w0[i] = s_i_w0_top5_mean
                s_top5_w0[i] = np.max(s_i_w0)
            
            # Lowest 5% wages with w = 1
            w1 = w_i[w_i == 1]
            if len(w1) == 0:
                s_low5_w1[i] = np.nan
            else:
                s_i_w1 = s_i[w1]
                s_i_w1_sort = np.sort(s_i_w1)
                n_in_low5 = int(np.max((0.05*len(s_i_w1_sort), 1)))
                s_i_w1_low5 = s_i_w1_sort[: n_in_low5]
                s_i_w1_low5_mean = np.mean(s_i_w1_low5)
                # s_low5_w1[i] = s_i_w1_low5_mean
                s_low5_w1[i] = np.min(s_i_w1)
        
        fig, (ax, ax_mu) = plt.subplots(nrows=2)
        ax.plot(time_values, s_top5_w0, "-", c="crimson", label=r"Top $5\%$ wages at $w=0$", alpha=0.8)
        ax.plot(time_values, s_low5_w1, "-", c="dodgerblue", label=r"Lowest $5\%$ wages at $w=1$", alpha=0.6)
        ax.axhline(y=self.s_min, linestyle="--", c="grey", alpha=0.9, label=r"$s_\text{min}$")
        ax.set(xlabel="Time [a.u.]", ylabel="Log Price [a.u.]", yscale="log")
        ax.grid()
        ax.legend(frameon=False)

        ax_mu.plot(time_values, mu/self.W, label=r"$P / W$", alpha=1)
        ax_mu.set(ylabel="Log Price [a.u.]", yscale="log")
        ax.grid()
        ax.legend(frameon=False)
        
        # Text, save, show
        self._text_save_show(fig, ax, "smin_effective", xtext=0.05, ytext=0.95, fontsize=1)


    def compare_smin_lifespan(self, group_name_list):
        # Get data
        lifespan_list = [self._get_lifespan(name) for name in group_name_list]
        labels = ["$s_{{\\min}}=10^{-9}$", "$s_{{\\min}}=10^{-5}$", "$s_{{\\min}}=10^{-3}$", "$s_{{\\min}}=10^{-2}$", ]  #[f"$s_{{\\min}}={name}$" for name in group_name_list]  # Clean LaTeX-style labels

        # Store LaTeX table rows
        table_rows = []

        for i in range(len(lifespan_list) - 1):
            for j in range(i + 1, len(lifespan_list)):
                res = scipy.stats.cramervonmises_2samp(lifespan_list[i], lifespan_list[j])
                row = f"{labels[i]} vs {labels[j]} & {res.statistic:.4f} & {res.pvalue:.4g} \\\\"
                table_rows.append(row)

        # Print LaTeX table
        print(r"""\begin{table}[ht]
            \centering
            \caption{Cramér--von Mises test comparing company lifespan distributions across $s_{\min}$ values.}
            \begin{tabular}{lcc}
            \hline
            Comparison & CvM statistic & $p$-value \\
            \hline""")

        for row in table_rows:
            print(row)

        print(r"""\hline
            \end{tabular}
            \end{table}""")


    def plot_lifespan_multiple_smin(self, group_name_list, labels=["$s_{{\\min}}=10^{-8}m$", "$s_{{\\min}}=10^{-1}m$", "$s_{{\\min}}=m$", "$s_{{\\min}}=2m$"], xlim=None):
        
        fig, ax = plt.subplots()
        c_list = [
    "deepskyblue",     # bright cyan-blue
    "mediumseagreen",  # soft green
    "goldenrod",       # mustard yellow
    "indianred",       # warm muted red
    "slateblue",       # soft purple-blue
    "coral",           # soft orange-pink
    "darkcyan",        # deep teal
    "darkkhaki",       # dusty yellow
    "mediumorchid",    # magenta-violet
    "steelblue",       # muted mid-blue
    "rosybrown",       # earthy pinkish-gray
    "peru"             # rich tan/brown-orange
]  
        # alpha_list = [0.9, 0.8, 0.6, 0.4]
        alpha_list = np.linspace(0.3, 0.9, len(group_name_list))[::-1]
        def _plotter(idx):
            # ax = ax_flat[idx]
            gname = group_name_list[idx]
            lifespan = self._get_lifespan(gname)
            # Bin and plot
            bins = np.arange(0, np.max(lifespan), 1)
            ax.hist(lifespan, bins=bins, density=True, color=c_list[idx], alpha=alpha_list[idx], label=labels[idx])
            ax.grid()
        
        for i in range(len(group_name_list)):
            _plotter(i)
        
        # Axis setup
        ax.set(ylabel="Empirical PMF", xlabel="Company lifespan",  yscale="log", xlim=xlim,)
        ax.legend(frameon=False, loc="upper right", fontsize=16)
        ax.grid()
        # y_ticks = [0, 0.01, 0.02, 0.03, 0.04]
        # y_ticklabels = y_ticks[::2]
        # self._axis_ticks_and_labels(ax, y_ticks=y_ticks, y_labels=y_ticklabels)
        xlim = ax.get_xlim()
        x_ticks = np.linspace(0, xlim[-1], 5).astype(np.int32)
        # x_ticks = [1, 500, 1000, 1500, 2000]
        x_ticklabels = x_ticks[::2]
        self._axis_ticks_and_labels(ax, x_ticks=x_ticks, x_labels=x_ticklabels, x_dtype="int")
        
        # Text, save, show
        self._text_save_show(fig, ax, "smin_lifespan", xtext=0.05, ytext=0.95, fontsize=1)

        
    def plot_running_KDE_multiple_s_min(self, group_name_list, bandwidth_s=None, eval_points=100, s_lim=None, kernel="gaussian", show_mean=False, show_title=False, suffix_list=None):
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
                ax.plot(time_values, mu_arr[index] / self.W, c="magenta", label=r"$P / W$", alpha=1, lw=0.6)

            # Axis and ticks
            self._axis_ticks_and_labels(ax, x_ticks, y_ticks, x_tick_labels, y_tick_labels, x_dtype="int")
            self._axis_labels_outer(ax, x_label="Time", y_label="Wage")

        # Add a single colorbar for all subplots, stretch it to the full height of the figure
        cbar = fig.colorbar(im, ax=axs, orientation='vertical', ticks=[], pad=0.01)# fraction=0.02, pad=0.04)
        cbar.set_label(label="Frequency", fontsize=label_fontsize)
        cbar.ax.tick_params(labelsize=0)  # Remove tick labels

        # Subplot label
        for i, axis in enumerate(axs.flatten()):
            self._subplot_label(axis, i, color="white", outline_color="darkred")

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
        elif data_name == "capital_individual_mean" or "capital_individual_all":
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
                ax.plot(time_values, mu_arr[index] / W_arr[index], c="magenta", label=r"$P / W$", alpha=1, lw=0.6)
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

            ax_time.plot(time_values, mu / self.W, label=r"$P/W$", c=self.colours["mu"])
            ax_time.set(xlabel="Time", ylabel="Price", title=r"Mean capital and $P$")
            ax_time.grid()
            ax_time.tick_params(axis="y", labelcolor=self.colours["mu"])    
            ax_time.set_ylabel(r"$P/W$", color=self.colours["mu"])

        if show_distributions:        
            # Return distributions
            ax_return.hist(bins[:-1], bins, weights=counts_mu, color=self.colours["mu"], label=r"$P$", alpha=0.95, density=True)
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
        ax_aggr.hist(bins[:-1], bins, weights=counts_mu, color=self.colours["mu"], label=r"$P$", alpha=0.95, density=True)
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


    def plot_return_individual_and_aggregate(self, Nbins_agg=None, Nbins_indi=None, ylim=None, show_aggregate=False, xticks=[-2, 0, 2]):
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
        if show_aggregate:
            nrows, ncols = 2, 1
            figsize = (10, 5)
        else:
            nrows, ncols = 1, 1
            figsize = (10, 2.5)
            
            
        fig, axs = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols)
        if show_aggregate:
            ax_aggr, ax_indi = axs
        else:
            ax_indi = axs

        def _plot_aggr():
            # ax_aggr: Aggregate distribution
            ax_aggr.hist(agg_bins[:-1], agg_bins, weights=counts_agg, color=self.colours["capital"], alpha=1, density=True)
            ax_aggr.set(ylabel=f"Prob. Density", yscale="log", ylim=ylim)
            ax_aggr.grid()
            # Ticks
            agg_x_ticks = np.round(np.linspace(edges_agg[0], edges_agg[-1], 5), 1)
            agg_x_ticklabels = agg_x_ticks[::2]
            self._axis_ticks_and_labels(ax_aggr, x_ticks=agg_x_ticks, x_labels=agg_x_ticklabels, y_ticks=[], y_labels=[])
            self._axis_log_ticks_and_labels(ax_aggr, exponent_range=np.log10(ylim), labels_skipped=1)
            
        def _plot_indi():        
            """ax_indi: Individual capital distributions"""
            # Ticks. First set the x ticks, then the log y ticks
            indi_x_ticks = xticks # np.floor(np.linspace(edges_indi[0], edges_indi[-1], 5))
            indi_x_ticklabels = indi_x_ticks #[f"{x:.1f}" for x in indi_x_ticks]
            # Histogram
            ax_indi.hist(r_indi, bins=Nbins_indi, color=self.colours["capital"],  alpha=1, density=True)
            ax_indi.set(xlabel="Return", yscale="log", ylim=ylim, ylabel="Prob. Density", xlim=(1.1*indi_x_ticks[0], 1.1*indi_x_ticks[-1]),)
            ax_indi.grid()
            
            self._axis_ticks_and_labels(ax_indi, x_ticks=indi_x_ticks, x_labels=indi_x_ticklabels, y_ticks=[], y_labels=[])            
            self._axis_log_ticks_and_labels(ax_indi, exponent_range=np.log10(ylim), labels_skipped=1)
            
        
        if show_aggregate:
            _plot_aggr()
            _plot_indi()   
        else:
            _plot_indi()
        
        # Text, save, show
        self._text_save_show(fig, ax_indi, "return_individual_and_aggregate", xtext=0.05, ytext=0.85, fontsize=0)
        

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
        s, mu, C = self._skip_values(self.s, self.mu, -self.d)
        mean_diff = mu / self.W - s.mean(axis=0)    

        # Create figure
        fig, (ax, ax_C, ax_means, ax_div) = plt.subplots(figsize=(10, 10), nrows=4, gridspec_kw={'height_ratios': [2, 1, 1, 1]}, sharex=True)
        label_fontsize = 20
        ticks_fontsize = 12.5
        
        # Wage
        im = ax.imshow(KDE_prob, aspect="auto", origin="lower", extent=[self.skip_values, self.time_steps, np.min(s_eval), np.max(s_eval)], cmap="hot")
        if show_mean: ax.plot(time, mu/self.W, c="magenta", lw=0.6)
        ax.set_ylabel("Wage [$]", fontsize=label_fontsize)
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
        self._axis_ticks_and_labels(ax, x_ticks=time_ticks, y_ticks=s_ticks, x_labels=time_ticklabels, y_labels=s_ticklabels)
        ax.set_xticklabels([])

        # Capital
        cbar_title = "Frequency"
        
        # Normalization
        # Do not differentiate between the top 100 minus percentile_cut of the values
        KDE_prob_C = np.clip(KDE_prob_C, 0, np.percentile(KDE_prob_C, percentile_cut))
        
        im_C = ax_C.imshow(KDE_prob_C, aspect="auto", origin="lower", extent=[self.skip_values, self.time_steps, np.min(C_eval), np.max(C_eval)], cmap=C_cmap)
        ax_C.set_ylabel("Capital [$]", fontsize=label_fontsize)
        
        # Add colorbar
        cbar_C = fig.colorbar(im_C, ax=ax_C, ticks=[], pad=-0.1, aspect=15)
        cbar_C.set_label(label=cbar_title, fontsize=label_fontsize)
        # Ticks
        C_ticks = np.linspace(0, C_lim[1], 5)
        C_ticklabels = [C_ticks[0], C_ticks[0]+(C_lim[1]-C_ticks[0])/2, C_lim[1]]
        self._axis_ticks_and_labels(ax_C, x_ticks=time_ticks, y_ticks=C_ticks, x_labels=time_ticklabels, y_labels=C_ticklabels)
        ax_C.set_xticklabels([])
        
        # Mean wage and capital
        twin_C = ax_means.twinx()
        twin_C.plot(time, C.mean(axis=0), c=self.colours["capital"])
        twin_C.set(xlim=self.xlim)
        twin_C.set_ylabel(r"$\bar{C}(t)$ [\$]", fontsize=label_fontsize, color=self.colours["capital"])
        twin_C.tick_params(axis="y", which="both", labelsize=ticks_fontsize, labelcolor=self.colours["capital"])
        
        ax_means.plot(time, mu/self.W, color=self.colours["mu"])
        ax_means.set_ylabel(r"$P / W$", fontsize=label_fontsize, color=self.colours["mu"])
        ax_means.tick_params(axis="y", which="both", labelsize=ticks_fontsize, labelcolor=self.colours["mu"])
        ax_means.grid()
        
        # Diversity and mean diff
        # Diversity
        twinx = ax_div.twinx()
        twinx.plot(time, diversity, c=self.colours["diversity"])
        twinx.set(xlim=self.xlim, ylim=(0, self.N))
        twinx.set_ylabel("Company diversity", fontsize=label_fontsize, color=self.colours["diversity"])
        twinx.tick_params(axis='y', which='major', labelsize=ticks_fontsize, labelcolor=self.colours["diversity"])
        # Ticks
        div_ticks = np.linspace(0, self.N, 5)
        div_ticklabels = [0, self.N//2, self.N]
        self._axis_ticks_and_labels(twinx, x_ticks=time_ticks, y_ticks=div_ticks, x_labels=time_ticklabels, y_labels=div_ticklabels, y_dtype="int")
        twinx.tick_params(axis='y', labelcolor=self.colours["diversity"], colors=self.colours["diversity"], which="both")
        
        # Mean diff
        mean_ticks = np.round(np.linspace(0, mean_diff.max(), 5), 2)
        mean_ticklabels = mean_ticks[::2]

        ax_div.plot(time, mean_diff, c=self.colours["mu"], label=r"$P / W - \bar{s}$", alpha=0.9)
        ax_div.grid()
        ax_div.set_ylabel(r"$P / W - \bar{s} $ [\$]", color=self.colours["mu"], fontsize=label_fontsize)
        ax_div.set_xlabel("Time [a.u.]", fontsize=label_fontsize)
        ax_div.set(xlim=self.xlim, ylim=(None, mean_ticks[-1]*1.1))
        ax_div.tick_params(axis='y', labelcolor=self.colours["mu"], labelsize=ticks_fontsize)
        # Ticks
        self._axis_ticks_and_labels(ax_div, x_ticks=time_ticks, y_ticks=mean_ticks, x_labels=time_ticklabels, y_labels=mean_ticklabels, x_dtype="int")
        ax_div.tick_params(axis='y', labelcolor=self.colours["mu"], colors=self.colours["mu"], which="both")

        # Subplot labels
        axis_all = (ax, ax_C, ax_means, ax_div)
        outline_color_list = ["black", "black", None, None]
        color_list = ["white", "white", "black", "black"]
        for i, axis in enumerate(axis_all):
            self._subplot_label(axis, i, fontsize=18, outline_color=outline_color_list[i], color=color_list[i])
        
        # Text, save, show
        self._text_save_show(fig, ax, "wage_capital_density_average_income", xtext=0.05, ytext=0.95, fontsize=0)
        
        
        
    def wage_density_diversity_increase_decrease(self, group_name_list, bandwidth_s, eval_points, kernel, s_lim, window_size=1) -> None:
        """The big plot that has KDE and diversity for multiple datasets, and below that ds vs frequency.

        Args:
            group_name_list (_type_): _description_
            time_scale_group_name_tensor (_type_): _description_
            bandwidth_s (_type_): _description_
            eval_points (_type_): _description_
            kernel (_type_): _description_
            s_lim (_type_): _description_
        """
        # Get the wage density (KDE), Diversity and increase-decrease data
        # Create empty arrays
        self._get_data(group_name_list[0])
        KDE_arr = np.zeros((len(group_name_list), eval_points, self.time_steps-self.skip_values))
        diversity_arr = np.zeros((len(group_name_list), self.time_steps-self.skip_values))
        salary_means_diff_arr = np.zeros((len(group_name_list), self.time_steps-self.skip_values))  
        increase_arr = np.zeros((len(group_name_list), self.time_steps-self.skip_values - 1))  # C diff reduces by 1
        decreased_arr = np.zeros_like(increase_arr)
        zero_workers_arr = np.zeros_like(increase_arr)
        N_arr = np.zeros(len(group_name_list))
                
        for i, gname in enumerate(group_name_list):
            # Load data and store it in arrays
            s_eval, KDE_prob = self.running_KDE("salary", bandwidth_s, eval_points, kernel, s_lim, gname=gname)
            time, diversity = self._worker_diversity(gname)
            s, mu, C, w = self._skip_values(self.s, self.mu, -self.d, self.w)
            w = w[:, :-1]
            salary_means_diff = mu / self.W - np.mean(s, axis=0)
            
            # Increase decrease
            C_diff = np.diff(C, axis=1)
            increased = np.count_nonzero(C_diff>0, axis=0)
            decreased = np.count_nonzero(C_diff<0, axis=0)
            zero_workers = np.count_nonzero(w == 0, axis=0)
            # Calculate rolling averages
            increased = uniform_filter1d(increased, size=window_size)
            decreased = uniform_filter1d(decreased, size=window_size)
            zero_workers = uniform_filter1d(zero_workers, size=window_size)
            
            # Store values
            KDE_arr[i, :, :] = KDE_prob
            diversity_arr[i, :] = diversity
            salary_means_diff_arr[i, :] = salary_means_diff
            increase_arr[i, :] = increased
            decreased_arr[i, :] = decreased
            zero_workers_arr[i, :] = zero_workers
            N_arr[i] = self.N
        
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
        # Last row 
        ax40 = fig.add_subplot(gs[4, 0])
        ax41 = fig.add_subplot(gs[4, 1])

        # Ticks
        tick_width, tick_width_minor = 1.8, 1
        tick_length, tick_length_minor = 6, 3
        
        time_first, time_last = self.time_values[self.skip_values] + 50, self.time_values[-1] - 50 + 1
        time_ticks = np.linspace(time_first, time_last, 5)
        time_labels = [time_first, time_first+(time_last-time_first)/2, time_last]
        
        KDE_min, KDE_max = 0, s_lim[-1]
        KDE_y_ticks = np.linspace(KDE_min, KDE_max, 5)
        KDE_y_labels= [KDE_min, KDE_min+(KDE_max-KDE_min)/2, KDE_max]
        
        means_min, means_max = np.min(salary_means_diff_arr), np.round(np.max(salary_means_diff_arr), 1)  # mu diff ticks
        print("Minimum of the average profit: ")
        print(np.min(salary_means_diff_arr))
        div_means_ticks = np.linspace(0, means_max, 5)  # Want zero as one of the ticks
        div_means_labels = [0, np.round((means_max-means_min)/2, 1), means_max]  
        
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
            N = N_arr[i]
            
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
                ax_KDE.set_ylabel("Wage [$]")
            else:
                ax_KDE.set_yticklabels([])
            # Remove all x ticks labels
            ax_KDE.set_xticklabels([])
            ax_KDE.tick_params(axis="both", width=tick_width, length=tick_length, which="major")
            ax_KDE.tick_params(axis="both", width=tick_width_minor, length=tick_length_minor, which="minor")
            
            # Diversity and mean salary difference
            # Mean salary difference
            ax_div.plot(time, s_diff, c=self.colours["mu"], label=r"$P / W - \hat{s}$ [\$]", alpha=0.9)
            ax_div.set(xlim=self.xlim, ylim=(means_min, means_max))
            ax_div.grid(which='both')
            self._axis_ticks_and_labels(ax_div, x_ticks=time_ticks, y_ticks=div_means_ticks, x_labels=time_labels, y_labels=div_means_labels, x_dtype="int")

            # Add y axis labels, remove y tick labels form second column
            if is_first_column:
                ax_div.set_ylabel(r"$P / W - \hat{s}$ [\$]", color=self.colours["mu"])
            else:
                ax_div.set_yticklabels([])
                
            # Add colour to the y tick labels
            ax_div.tick_params(axis='y', labelcolor=self.colours["mu"], colors=self.colours["mu"], which="both")
            ax_div.tick_params(axis="both", width=tick_width, length=tick_length, which="major")
            ax_div.tick_params(axis="both", width=tick_width_minor, length=tick_length_minor, which="minor")

            # Diversity on twin axis
            twinx = ax_div.twinx()
            twinx.plot(time, diversity, label="Company diversity", c=self.colours["diversity"], alpha=0.7)
            twinx.set(ylim=(0, N))  
            # Ticks
            div_y_ticks = np.linspace(0, N, 5)
            div_y_labels = [0, N//2, N]
            self._axis_ticks_and_labels(twinx, x_ticks=time_ticks, y_ticks=div_y_ticks, x_labels=time_labels, y_labels=div_y_labels, x_dtype="int", y_dtype="int")
            twinx.tick_params(axis='y', labelcolor=self.colours["diversity"], colors=self.colours["diversity"], which="both")
            twinx.tick_params(axis="both", width=tick_width, length=tick_length, which="major")
            twinx.tick_params(axis="both", width=tick_width_minor, length=tick_length_minor, which="minor")
            # Add y axis labels if second column, remove y tick labels from first row
            if not is_first_column:
                twinx.set_ylabel("Company diversity", color=self.colours["diversity"])
            else:
                twinx.set_yticklabels([])
                
            # Remove x tick labels from the first row. Add time labels to the second row
            twinx.set_xticklabels([])
            ax_div.set_xticklabels([])

        # Increase - decrease plot
        # Get data then do the plotting
        y_lim_increase_decrease = (0, 0.5)
        increased_40 = increase_arr[-2, :] / self.N
        decreased_40 = decreased_arr[-2, :] / self.N
        zero_workers_40 = zero_workers_arr[-2, :] / self.N
        increased_41 = increase_arr[-1, :] / self.N
        decreased_41 = decreased_arr[-1, :] / self.N
        zero_workers_41 = zero_workers_arr[-1, :] / self.N
        time_minus_1 = time[:-1]
        c_increase, c_w0, c_decrease = "limegreen", "royalblue", "crimson"
        
        ax40.plot(time_minus_1, increased_40, "-", color=c_increase, alpha=0.7, label=r"$\Delta C_k > 0$")
        ax40.plot(time_minus_1, decreased_40, "-", color=c_decrease, alpha=0.7, label=r"$\Delta C_k < 0$")
        ax40.plot(time_minus_1, zero_workers_40, "-", color=c_w0, alpha=0.7, label=r"$w=0$")
        ax40.set(xlabel="Time [a.u.]", ylabel="Fraction", xlim=self.xlim, ylim=y_lim_increase_decrease)
        ax40.grid()        
        
        ax41.plot(time_minus_1, increased_41, "-", color=c_increase, alpha=0.7)
        ax41.plot(time_minus_1, decreased_41, "-", color=c_decrease, alpha=0.7)
        ax41.plot(time_minus_1, zero_workers_41, "-", color=c_w0, alpha=0.7)
        ax41.set(xlabel="Time [a.u.]", xlim=self.xlim, ylim=y_lim_increase_decrease)
        ax41.grid()
        
        # Ticks for increase decrease
        y_ticks_increase_decrease = [y_lim_increase_decrease[0], y_lim_increase_decrease[0]+(y_lim_increase_decrease[1] - y_lim_increase_decrease[0])/2, y_lim_increase_decrease[1]]
        self._axis_ticks_and_labels(ax40, x_ticks=time_ticks, y_ticks=y_ticks_increase_decrease, x_labels=time_labels, y_labels=y_ticks_increase_decrease, x_dtype="int")
        self._axis_ticks_and_labels(ax41, x_ticks=time_ticks, y_ticks=y_ticks_increase_decrease, x_labels=time_labels, y_labels=y_ticks_increase_decrease, x_dtype="int")
        ax41.set_yticklabels([])
        
        
        # After plotting on your two subplots (say, ax_left and ax_right)
        # handles, labels = ax40.get_legend_handles_labels()

        # Create a figure-level legend:
        # fig.legend(handles, labels,
        #         loc='upper center',         # location relative to the figure
        #         bbox_to_anchor=(0.5, 0.19),   # adjust this tuple to position the legend
        #         ncol=3,)                     # number of columns in the legend)
        
        # Colorbar
        # Add a new axis to the figure for the colorbar
        cbar_ax = fig.add_axes([0.15, 0.97, 0.25, 0.02])  # figure space [left, bottom, width, height]
        # Create a horizontal colorbar using the dedicated axis
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', ticks=[])
        cbar.outline.set_edgecolor('white')
        # cbar.ax.xaxis.set_label_position('top')  # move the label to the top
        # cbar.set_label(label="Frequency", fontsize=15)
        
        # Text save show
        self._text_save_show(fig, ax00, "parameters", xtext=0.05, ytext=0.95, fontsize=0)
        
        
    def plot_worker_distribution(self, Nbins=None, xlim=None, ylim=(1e-5, 2e0), x_ticks=None, bars_or_points="bars", p0=(1.059, 1), w_lim_fit=(None, None)):
        """Company Size. Histogram of the counts of companies with workers.
        """
        assert bars_or_points in ["bars", "points"]
        
        # Get data and skip values
        self._get_data(self.group_name)
        w = self._skip_values(self.w)
        
        w = w[w>0]

        def power_law(x, a, b):
            return b * x ** (-a)
        
        # Bin it
        if Nbins is None:
            Nbins = int(np.sqrt(w.size))

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Binning
        bins = 10 ** np.linspace(np.log10(1e-1), np.log10(np.max(w) * 10), Nbins)  # Log x cuts off large values if max range value is not increased
        counts, edges = np.histogram(w, bins=bins, density=True)
        x_plot = (edges[1] - edges[0]) / 2 + edges[:-1]
        ax.set(xscale="log")
        if bars_or_points == "bars":
            ax.stairs(counts, edges, color=self.colours["workers"], alpha=1)
        elif bars_or_points == "points":
            ax.plot(x_plot, counts, ".", color=self.colours["workers"], label="Data")
        
        # Plot the empirical results (a line with slope -2)
        x_line = np.linspace(x_plot[0], x_plot[-1], 250)
        y_emp = power_law(x_line, a=2.059, b=1)
        ax.plot(x_line, y_emp, "k-", label="Empirical, slope=2")
        
        # Setup
        ax.grid()
        ax.set(xlabel="Number of workers", ylabel="Prob. Density", yscale="log", xlim=xlim, ylim=ylim)
        # y ticks        
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
        
        # Skip values, remove w=0 companies
        w = self._skip_values(self.w)
        w = w[w>0]
        w = w.ravel()
        
        # KDE
        x_eval, P = self.onedim_KDE(w, bandwidth, x_eval=x_eval, N_eval_points=N_eval_points, kernel="gaussian", log_scale=log_xscale)
        
        fig, ax = plt.subplots(figsize=(10, 5)) 
        ax.plot(x_eval, P, c=self.colours["workers"])
        ax.set(xlabel="Number of workers", ylabel="Prob. Density", yscale="log", xlim=xlim, ylim=ylim)
        if log_xscale:
            ax.set(xscale="log")
        ax.grid()
        
        # Text save show
        self._text_save_show(fig, ax, "worker_KDE", xtext=0.05, ytext=0.85, fontsize=0)
    
    
    def plot_worker_KDE_over_time(self, bandwidth, N_eval_points=600, kernel="gaussian", w_lim=None, percentile_cut=100):
        # Get data
        w_eval, KDE_prob = self.running_KDE("workers", bandwidth, N_eval_points, kernel, w_lim)  # KDE probabilities
        
         # Do not differentiate between the top 100 minus percentile_cut of the values
        KDE_prob = np.clip(KDE_prob, 0, np.percentile(KDE_prob, percentile_cut))
        
        # Create figure        
        fig, ax_KDE = plt.subplots(figsize=(10, 5))
        im = ax_KDE.imshow(KDE_prob, cmap="plasma", aspect="auto", origin="lower", extent=[self.skip_values, self.time_steps, np.min(w_eval), np.max(w_eval)])
        
        ax_KDE.set(xlabel="Time", ylabel="Workers")
        
        cbar = fig.colorbar(im, ax=ax_KDE, ticks=[], pad=0.01, aspect=30)
        cbar.set_label(label="Frequency", fontsize=15)
        # Text, save, show
        self._text_save_show(fig, ax_KDE, "worker_KDE_over_time", xtext=0.05, ytext=0.95, fontsize=0)


    def plot_worker_four_different_times(self, time_list=None, time_points=20, Nbins=None, xlim=None, ylim=None, bars_or_points="bars", slope_offset=1):
        # Get data and skip values
        self._get_data(self.group_name)
        
        w = self._skip_values(self.w)

        def _slope(w_max):
            x_slope = np.linspace(1, w_max, 200)
            slope = 2.059
            y_slope = (slope_offset/x_slope)**slope
            return x_slope, y_slope
                    
        # Create fig
        fig, axs = plt.subplots(figsize=(10, 5), ncols=2, nrows=2)
        axs = axs.flatten()
        # Loop over data and plot each
        for i, T in enumerate(time_list):
            # Get axis and data
            ax = axs[i]
            w_T = w[:, T: T + time_points]
            w_T = w_T[w_T>0]

            # Bin data
            if Nbins is None:
                Nbins = int(np.sqrt(len(w_T)))
            counts, edges = np.histogram(w_T, bins=Nbins, density=True)
            
            # Plot        
            if bars_or_points == "bars": 
                ax.stairs(counts, edges, color=self.colours["workers"], alpha=1, label="Data")
            else:
                x_plot = (edges[1] - edges[0]) / 2 + edges[:-1]
                ax.plot(x_plot, counts, ".", label="Data", c=self.colours["workers"])
            
            # Plot empirical data: line with slope -2.059
            x_slope, y_slope = _slope(w_T.max())
            ax.plot(x_slope, y_slope, "-", label="Empirical")    
            ax.set(xlabel="Workers", xscale="log", yscale="log", ylabel="Prob. Density", xlim=xlim, ylim=ylim)
            ax.text(x=0.1, y=0.9, s=fr"$T=[{self.skip_values+T}, {self.skip_values+T+time_points}]$", transform=ax.transAxes, horizontalalignment="left")
        
        # Text, save, show
        self._text_save_show(fig, ax, "worker_four_different_times", xtext=0.05, ytext=0.95, fontsize=0)
        
        
    def plot_increased_decreased(self, window_size=5, bandwidth_s=0.004, s_lim=(0.000, 0.18), eval_points=400, kernel="epanechnikov", percentile_cut=100):
        """Plot the number of companies who increased vs decreased their wages together with the wage density
        """
        # Get data
        self._get_data(self.group_name)
        # KDE
        s_eval, KDE_prob = self.running_KDE("salary", bandwidth_s, eval_points, kernel, s_lim)  # KDE probabilities
        KDE_prob = np.clip(KDE_prob, 0, np.percentile(KDE_prob, percentile_cut))

        # Calculate capital diff
        s, C, time, w = self._skip_values(self.s, -self.d, self.time_values[:-1], self.w)
        w = w[:, :-1]
        
        C_diff = np.diff(C, axis=1)
        increased = np.count_nonzero(C_diff>0, axis=0)
        increased_at_zero = np.count_nonzero(C_diff==0, axis=0)
        decreased = np.count_nonzero(C_diff<0, axis=0)
        zero_workers = np.count_nonzero(w == 0, axis=0)
        increased_at_zero -= zero_workers
        # Calcluate whether increased or not by looking at salary differences
        s_diff = np.diff(s, axis=1)
        # increased = np.count_nonzero(s_diff>0, axis=0)
        # decreased = np.count_nonzero(s_diff<=0, axis=0)
        
        # Calculate rolling averages
        increased = uniform_filter1d(increased, size=window_size)
        increased_at_zero = uniform_filter1d(increased_at_zero, size=window_size)
        decreased = uniform_filter1d(decreased, size=window_size)
        zero_workers = uniform_filter1d(zero_workers, size=window_size)
        
        # Create figure
        fig, (ax_w, ax) = plt.subplots(figsize=(10, 5), nrows=2)
        # KDE
        im = ax_w.imshow(KDE_prob, aspect="auto", origin="lower", extent=[self.skip_values, self.time_steps, np.min(s_eval), np.max(s_eval)], cmap="hot")
        ax_w.set(ylabel="Wage")
        
        # Increase/Decrease
        ax.plot(time, increased, "-", color="green", label=r"$\Delta C > 0$")
        ax.plot(time, decreased, "--", color="red", label=r"$\Delta C < 0$")
        # ax.plot(time, increased_at_zero, ls="dashdot", color="orange", label=r"$\Delta C = 0$")
        ax.plot(time, zero_workers, ls=(0, (3, 1, 1, 1)), color="blue", label=r"$w=0$")
        title = f"Mean decrease = {np.mean(decreased):.1f}, Mean increase = {np.mean(increased):.1f}"
        ax.set(xlabel="Time", ylabel="Number of companies", xlim=self.xlim, title=title)
        ax.grid()
        
        self._add_legend(ax, ncols=4, fontsize=8, x=0.5, y=0.85)
        
        # Text save show
        self._text_save_show(fig, ax, "increased_decreased", xtext=0.05, ytext=0.85, fontsize=4)
        
        
    def plot_mu_bankruptcy_correlation(self, time_shift=0):
        # Get data
        self._get_data(self.group_name)
        mu, bankrupt = self._skip_values(self.mu/self.W, self.went_bankrupt/self.N)
        
        # Shift data
        title = ""
        if time_shift != 0:
            mu = mu[:-time_shift]
            bankrupt = bankrupt[time_shift:]
            title = f"Shift = {time_shift}"
        
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(mu, bankrupt, ".", alpha=0.8, markersize=2)
        ax.set(xlabel=r"$P$", ylabel="Fraction bankrupt", title=title)
        ax.grid()
        
        self._text_save_show(fig, ax, "mu_bankruptcy_correlation", xtext=0.05, ytext=0.85, fontsize=4)
    
    
    def plot_recession_variables(self, window_size=1):
        """Different variables for finding recessions
        """
        # Get data
        self._get_data(self.group_name)
        
        mu, C, w, s, time = self._skip_values(self.mu, -self.d, self.w, self.s, self.time_values)
        
        # Average profit
        average_profit = mu / self.W - s.mean(axis=0)
        
        # Increase decrease
        C_diff = np.diff(C, axis=1)
        increased = np.count_nonzero(C_diff>0, axis=0)
        decreased = np.count_nonzero(C_diff<0, axis=0)
        zero_workers = np.count_nonzero(w == 0, axis=0)
        net_increase_decrease = increased - decreased
        
        # Apply rolling average 
        mu = uniform_filter1d(mu, size=window_size)
        net_increase_decrease = uniform_filter1d(net_increase_decrease, size=window_size)
        zero_workers = uniform_filter1d(zero_workers, size=window_size)
        average_profit = uniform_filter1d(average_profit, size=window_size)
        
        fig, ax_arr = plt.subplots(ncols=1, nrows=4, figsize=(10, 10))
        # ax_mu = ax_arr[0, 0]        
        # ax_C_diff = ax_arr[1, 0]
        # ax_w0 = ax_arr[0, 1]        
        # ax_profit = ax_arr[1, 1]
        ax_mu, ax_profit, ax_w0, ax_C_diff = ax_arr
        
        # Plot        
        ax_mu.plot(time, mu / self.W, )
        ax_C_diff.plot(time[:-1], net_increase_decrease)
        ax_w0.plot(time, zero_workers)
        ax_profit.plot(time, average_profit, )
        
        # Axis setup
        ax_flat = ax_arr.flatten()
        for ax in ax_flat:
            ax.grid()
            ax.set(xlim=self.xlim)
        ax_mu.set(title=r"$P / W$", ylabel="Price")
        ax_C_diff.set(title=r"$N_{\Delta C > 0} - N_{\Delta C < 0}$", ylabel="Companies")
        ax_w0.set(title=r"$N_{w=0}$", ylabel="Companies")
        ax_profit.set(title=r"$P / W - \hat{s}$", ylabel="Price")
        
        fig.suptitle(f"Window size = {window_size}", fontsize=10)

        # Text save show
        self._text_save_show(fig, ax_mu, "recession", xtext=0.05, ytext=0.85, fontsize=2)
        
        
    def plot_recession(self, Nbins, Nbins_NBER, window_size=10, peak_distance=20, peak_width=5, peak_height=0.11, trough_height=-0.13, peak_prominence=0.01, trough_prominence=0.01, plot_peaks=False):
        # Get model data
        time_between, duration = self._recession_time_between_and_duration(window_size, peak_distance, peak_width, peak_height, trough_height, peak_prominence, trough_prominence, plot_peaks)
        
        # Get NBER data
        time_between_NBER, duration_NBER = self._load_recession_data()
        
        # Bin data
        if Nbins is None: 
            Nbins = int(np.sqrt(len(duration)))
        if Nbins_NBER is None: 
            Nbins_NBER = int(np.sqrt(len(time_between_NBER)))
        
        # Model    
        counts_time, edges_time = np.histogram(time_between, bins=Nbins, density=True)
        counts_duration, edges_duration = np.histogram(duration, bins=Nbins, density=True)
        ylim = (0, 1.05*np.max((counts_time.max(), counts_duration.max())))
        # NBER
        counts_time_NBER, edges_time_NBER = np.histogram(time_between_NBER, bins=Nbins_NBER, density=True)
        counts_duration_NBER, edges_duration_NBER = np.histogram(duration_NBER, bins=Nbins_NBER, density=True)
        ylim_NBER = (0, 1.05*np.max((counts_time_NBER.max(), counts_duration_NBER.max())))
        
        # create figure
        fig, ax_arr = plt.subplots(ncols=2, nrows=2, figsize=(10, 5))
        ax_time = ax_arr[0, 0]
        ax_duration = ax_arr[1, 0]
        ax_time_NBER = ax_arr[0, 1]
        ax_duration_NBER = ax_arr[1, 1]
        
        # Plot
        ax_time.hist(edges_time[:-1], edges_time, weights=counts_time, color=self.colours["time"])
        ax_duration.hist(edges_duration[:-1], edges_duration, weights=counts_duration, color=self.colours["time"])
        ax_time_NBER.hist(edges_time_NBER[:-1], edges_time_NBER, weights=counts_time_NBER, color=self.colours["time"])
        ax_duration_NBER.hist(edges_duration_NBER[:-1], edges_duration_NBER, weights=counts_duration_NBER, color=self.colours["time"])
        
        # Axis setup
        ax_time.set(xlabel="Time between recessions [a.u.]", ylabel="Prob. Density", ylim=ylim)
        ax_duration.set(xlabel="Recession duration [a.u.]", ylim=ylim)
        ax_time_NBER.set(xlabel="Time between recessions [days]", ylabel="Prob. Density", ylim=ylim_NBER)
        ax_duration_NBER.set(xlabel="Recession duration [days]", ylim=ylim_NBER)
        ax_time.grid()
        ax_duration.grid()
        ax_time_NBER.grid()
        ax_duration_NBER.grid()
        
        # Text save show
        self._text_save_show(fig, ax_time, "recession_distributions", xtext=0.05, ytext=0.85, fontsize=2)        
        
        
    def plot_lifespan(self, bin_width=None):
        # Get lifespan data
        lifespan = self._get_lifespan()
        
        # Bin it
        if bin_width is None: bin_width = 1
        bins = np.arange(1, np.max(lifespan), bin_width)
        counts, edges = np.histogram(lifespan, bins=bins, density=True)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(edges[:-1], edges, weights=counts, color=self.colours["time"])
        ax.set(xlabel="Company Lifespan", ylabel="Prob. Density", yscale="log")
        ax.grid()
        
        # Text save show
        self._text_save_show(fig, ax, "lifespan", xtext=0.05, ytext=0.85, fontsize=2)        


    def _apply_manual_ticks(self, ax):
        """
        3 majors + 2 minors on X (linear, no sci notation)
        2 majors + 1 minor on Y; if log scale, LaTeX style and label note.
        Also turns on grid for majors & minors.
        """
        # --- X axis ---
        xmin, xmax = ax.get_xlim()
        xmaj = np.linspace(xmin, xmax, 3)
        xminor = np.linspace(xmin, xmax, 5)[1:-1]

        ax.set_xticks(xmaj)
        ax.set_xticks(xminor, minor=True)
        ax.xaxis.set_minor_formatter(NullFormatter())

        ax.tick_params(axis="x", which="major", length=6, width=1.5)
        ax.tick_params(axis="x", which="minor", length=3, width=1)

        # --- Y axis ---
        ymin, ymax = ax.get_ylim()
        ymaj = [ymin, ymax]
        # midpoint: geometric if log, arithmetic if linear
        if ax.get_yscale() == "log":
            ymid = np.sqrt(ymin * ymax)
        else:
            ymid = 0.5*(ymin + ymax)
        yminor = [ymid]

        ax.set_yticks(ymaj)
        ax.set_yticks(yminor, minor=True)

        if ax.get_yscale() == "log":
            # LaTeX style: 10^k
            fmt = LogFormatterMathtext(base=10, labelOnlyBase=False)
            ax.yaxis.set_major_formatter(fmt)

        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="y", which="major", length=6, width=1.5)
        ax.tick_params(axis="y", which="minor", length=3, width=1)

        # --- Grids ---
        ax.grid(which="major", linestyle="-", linewidth=0.7, alpha=0.7)
        ax.grid(which="minor", linestyle="-", linewidth=0.5, alpha=0.7)



    def plot_economic_results(self, recession_parameters, 
                              Nbins_asset_return=None, ylim_asset_return=None, ylim_asset_return_data=None, Nbins_asset_return_data=None,
                              ylim_lifespan=None, bin_width_lifespan=1, Nbins_recession=None, Nbins_NBER_time=15, Nbins_NBER_duration=12, 
                              Nbins_size=None, xlim_size=None, ylim_size=None, SP500_change_days=10,
                              inflation_change_type="log", window_size_inflation=10, window_size_second_inflation=20, inflation_time_values=1000):
        """In the first coloumn, plot model results and in the second plot data.
        Plot:
            0. Asset return
            1. Company lifespan. 
            2. Reccesion duration
            3. Time between recessions
            4. Company size      
            5. Inflation  
        """
        # -- Get data --
        self._get_data(self.group_name)
        # Asset return
        asset_return = self._asset_return("capital_individual_all", time_period=1)
        asset_return_SP500 = self._load_sp500_asset_return(SP500_change_days) 
        
        # Company lifespan
        lifespan = self._get_lifespan()
        lifespan_data_x, lifespan_data_logy = self._load_lifespan_data()
        
        # Recession duration and time between
        try: 
            time_between_recessions, recession_duration, troughs_model, peaks_model = self._load_recession_results()
        except FileNotFoundError:
            print("No recession results found, generating results now")
            self._save_recession_results(**recession_parameters)
            time_between_recessions, recession_duration, troughs_model, peaks_model = self._load_recession_results()
        troughs_model = troughs_model[troughs_model < inflation_time_values]
        peaks_model = peaks_model[peaks_model < inflation_time_values]
        troughs_model += self.skip_values
        peaks_model += self.skip_values
        # Get NBER recession data data
        (time_between_NBER, duration_NBER), (time_between_NBER_PW, duration_NBER_PW) = self._load_recession_data(separate_post_war=True)
        df_recession_data = self._load_peak_trough_data()
        troughs_NBER = df_recession_data["trough"]
        peaks_NBER = df_recession_data["peak"]
        
        # Company size        
        w, time_for_inflation = self._skip_values(self.w, self.time_values[:self.skip_values+inflation_time_values])
        w = w[w > 0]
        # Company size data
        labels_size_data, counts_size_data = self._prepare_firm_size_pmf(normalize=True)
        
        # Inflation
        _, mu_smooth_inflation = self._get_inflation(change_type=inflation_change_type, window_size=window_size_inflation)
        mu_smooth_inflation = uniform_filter1d(mu_smooth_inflation, size=window_size_second_inflation)
        _, PCE_inflation = self._load_inflation_data(source="PCE", change_type=inflation_change_type, annualized=False)
        
        # Create figure and unpack axes
        fig, ax_arr = plt.subplots(figsize=(10, 10), ncols=2, nrows=6, gridspec_kw={'hspace': 0.15})
        ax_asset = ax_arr[0, :]
        ax_lifespan = ax_arr[1, :]
        ax_time_between = ax_arr[2, :]
        ax_duration = ax_arr[3, :]
        ax_size = ax_arr[4, :]
        ax_inflation = ax_arr[5, :]
        
        # Asset return
        # Gaussian to compare with
        x_gauss = np.linspace(-5, 5, 300)
        gauss = norm.pdf(x_gauss, loc=0, scale=np.std(np.ravel(asset_return[~np.isnan(asset_return)])))
        if Nbins_asset_return is None: Nbins_asset_return = int(0.9 * np.sqrt(len(asset_return)))
        ylim_asset_return = (1e-4, 5e0)
        ylim_asset_return_data = (1e-2, 1e2)
        asset_return_x_ticks = [-3, -1.5, 0, 1.5, 3] # np.floor(np.linspace(edges_indi[0], edges_indi[-1], 5))
        asset_return_x_ticklabels = asset_return_x_ticks[::2]  # {-3: "-3%", 0: "0%", 3: "3%"}
        ax_asset[0].hist(asset_return, bins=Nbins_asset_return, color=self.colours["capital"], density=True, label="Model results")
        ax_asset[0].plot(x_gauss, gauss, c="black", label="Gaussian", alpha=0.8)
        ax_asset[0].set(xlabel="Ln return", ylabel="PDF", ylim=ylim_asset_return, yscale="log", xlim=(-3.5, 3.5))
        ax_asset[0].grid()
        ax_asset[0].legend(frameon=False, loc="upper right")
        self._axis_ticks_and_labels(ax_asset[0], x_ticks=asset_return_x_ticks, x_labels=asset_return_x_ticklabels)
            
        # Asset return data
        gauss_data = norm.pdf(x_gauss, loc=0, scale=np.std(asset_return_SP500))
        ylim_asset_return_data = (3e-5, 2e1)
        if Nbins_asset_return_data is None: Nbins_asset_return_data = int(np.sqrt(len(asset_return_SP500)))
        ax_asset[1].hist(asset_return_SP500, bins=Nbins_asset_return_data, color=self.colours["capital"], density=True, label="S&P 500 Data")
        ax_asset[1].plot(x_gauss, gauss_data, c="black", label="Gaussian", alpha=0.8)
        ax_asset[1].set(xlabel="Ln return", ylim=ylim_asset_return_data, yscale="log", ylabel="PDF", xlim=(-2.05, 2.05))
        ax_asset[1].grid()
        ax_asset[1].legend(frameon=False, loc="upper right")
        asset_return_data_x_ticks = [-2., -1.25, 0, 1.25, 2.] 
        asset_return_data_x_ticklabels = asset_return_data_x_ticks[::2] #{-2.5: "-2.5%", 0: "0%", 2.5: "2.5%"}
        self._axis_ticks_and_labels(ax_asset[1], x_ticks=asset_return_data_x_ticks, x_labels=asset_return_data_x_ticklabels)
        ax_asset[1].yaxis.set_major_locator(LogLocator(numticks=3))

        # Lifespan
        # Define xticks for the three time results (lifespan, time between recession, recession duration)
        # Disregard the top 2% highest lifespan values in the max to prevent extreme outliers
        lifespan_sorted = np.sort(lifespan)
        lifespan_for_max = lifespan_sorted[: int(lifespan_sorted.size * 0.999)]
        time_max = np.max((lifespan_for_max[-1], time_between_recessions.max(), recession_duration.max()))
        time_min = 0
        xlim_time = (time_min, time_max)
        xtime_round = 50
        xlim_time = np.round(np.array(xlim_time) / xtime_round) * xtime_round
        
        if ylim_lifespan is None: ylim_lifespan = (4e-7, 8e-2)
        bins_life_span = np.arange(1, time_max, bin_width_lifespan)
        lifespan_counts, lifespan_edges = np.histogram(lifespan, bins=bins_life_span, density=True)
        ax_lifespan[0].hist(lifespan_edges[:-1], lifespan_edges, weights=lifespan_counts, color=self.colours["time"])
        ax_lifespan[0].set(xlabel="Company Lifespan [a.u.]", ylabel="PMF", yscale="log", ylim=ylim_lifespan, xlim=xlim_time)
        ax_lifespan[0].grid()
        # Create insert in the top right corner that shows the first values
        axins = inset_axes(ax_lifespan[0], "30%", "30%", loc="upper right")
        axins.hist(lifespan, bins=bins_life_span, color=self.colours["time"], density=True)
        insert_xlim = (2, 14)
        insert_y_possible = lifespan_counts[0 : insert_xlim[1]]
        insert_ylim = (insert_y_possible[insert_y_possible>0].min()*0.9, insert_y_possible.max()*1.1)
        axins.set_xlim(*insert_xlim)
        axins.set_ylim(*insert_ylim)
        axins.set(yscale="log", yticks=[])
        axins.set_xticks([2, 8, 14], [2, 8, 14], fontsize=6)
        axins.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
        mark_inset(ax_lifespan[0], axins, loc1=2, loc2=3, fc="none", ec="0.5")
        
        # Ticks
        x_ticks_time = np.linspace(xlim_time[0], xlim_time[1], 5)
        x_ticklabels_time = x_ticks_time[::2]
        self._axis_ticks_and_labels(ax_lifespan[0], x_ticks=x_ticks_time, x_labels=x_ticklabels_time, x_dtype="int")
        ax_lifespan[0].yaxis.set_major_locator(LogLocator(numticks=3))

        # Lifespan data
        ylim_lifespan_data = (5e-1, 5e3)
        lifespan_data_y = 10 ** lifespan_data_logy
        ax_lifespan[1].plot(lifespan_data_x, lifespan_data_y, ".", color=self.colours["time"])
        ax_lifespan[1].set(xlabel="Company Lifespan [years]", yscale="log", ylim=ylim_lifespan_data, ylabel="Counts")
        ax_lifespan[1].grid()
        data_x_ticks_lifespan = [0, 15, 30, 45, 60] 
        data_x_ticklabels_lifespan = data_x_ticks_lifespan[::2] 
        self._axis_ticks_and_labels(ax_lifespan[1], x_ticks=data_x_ticks_lifespan, x_labels=data_x_ticklabels_lifespan, x_dtype="int")
        ax_lifespan[1].yaxis.set_major_locator(LogLocator(numticks=3))
        
        # Recessions: Time between and duration
        day_to_year_factor = 1 / 365
        duration_NBER *= day_to_year_factor
        time_between_NBER *= day_to_year_factor
        duration_NBER_PW *= day_to_year_factor
        time_between_NBER_PW *= day_to_year_factor
        # Bin data
        if Nbins_recession is None: 
            Nbins = int(np.sqrt(len(recession_duration)))
        bins_time_NBER = np.linspace(time_between_NBER.min(), time_between_NBER.max(), Nbins_NBER_time)
        bins_duration_NBER = np.linspace(duration_NBER.min(), duration_NBER.max(), Nbins_NBER_duration)
                
        # Model
        counts_time, edges_time = np.histogram(time_between_recessions, bins=Nbins, density=True)
        counts_duration, edges_duration = np.histogram(recession_duration, bins=Nbins, density=True)
        # NBER
        counts_time_NBER, edges_time_NBER = np.histogram(time_between_NBER, bins=bins_time_NBER, density=False)  # All NBER data
        counts_duration_NBER, edges_duration_NBER = np.histogram(duration_NBER, bins=bins_duration_NBER, density=False)  # All NBER data
        counts_time_PW_NBER, edges_time_PW_NBER = np.histogram(time_between_NBER_PW, bins=bins_time_NBER)
        counts_duration_PW_NBER, edges_duration_PW_NBER = np.histogram(duration_NBER_PW, bins=bins_duration_NBER)
        
        ax_time_between[0].hist(edges_time[:-1], edges_time, weights=counts_time, color=self.colours["time"])
        ax_duration[0].hist(edges_duration[:-1], edges_duration, weights=counts_duration, color=self.colours["time"])
        ax_time_between[1].hist(edges_time_NBER[:-1], edges_time_NBER, weights=counts_time_NBER, color=self.colours["time"], label="Since 1854")  # All NBER data
        ax_duration[1].hist(edges_duration_NBER[:-1], edges_duration_NBER, weights=counts_duration_NBER, color=self.colours["time"])  # All NBER data
        
        # Post war NBER (recession)
        post_war_colour = "purple"
        post_war_alpha = 0.5
        ax_time_between[1].hist(edges_time_PW_NBER[:-1], edges_time_PW_NBER, weights=counts_time_PW_NBER, color=post_war_colour, alpha=post_war_alpha, label="Since 1946")
        ax_duration[1].hist(edges_duration_PW_NBER[:-1], edges_duration_PW_NBER, weights=counts_duration_PW_NBER, color=post_war_colour, alpha=post_war_alpha, label="Since 1946")
        
        # Legend for the NBER data and post war data
        colour_PW_blend = self._blend_colours(post_war_colour, self.colours["time"], post_war_alpha)
        legend_handles = [
            Patch(facecolor=self.colours["time"], label="Since 1854"),
            Patch(facecolor=colour_PW_blend, label="Since 1946")
        ]
        ax_time_between[1].legend(handles=legend_handles, loc="upper right", frameon=False, )
        ax_duration[1].legend(handles=legend_handles, loc="upper right", frameon=False, )
        
        # Axis setup
        xlim_NBER = (0, 13)
        ylim_time_between = (0, 1.05*np.max((counts_time.max(), counts_time_NBER.max())))
        ylim_duration = (0, 1.05*np.max((counts_duration.max(), counts_duration_NBER.max())))
        ax_time_between[0].set(xlabel="Time between recessions [a.u.]", ylabel="PMF", xlim=xlim_time)# ylim=ylim_time_between)
        ax_duration[0].set(xlabel="Recession duration [a.u.]", xlim=xlim_time, ylabel="PMF")#ylim=ylim_duration)
        ax_time_between[1].set(xlabel="Time between recessions [years]", ylabel="Counts", xlim=xlim_NBER)# ylim=ylim_time_between)
        ax_duration[1].set(xlabel="Recession duration [years]", ylabel="Counts", xlim=xlim_NBER)#ylim=ylim_duration)
        ax_time_between[0].grid()
        ax_duration[0].grid()
        ax_time_between[1].grid()
        ax_duration[1].grid()
        
        # Ticks
        yticks_time_between = np.array([0, 0.75, 1.5]) * 1e-3
        yticklabels_time_between = {0: r"$0$", 0.75e-3: r"$0.75 \times 10^{-3}$", 1.5e-3: r"$1.5 \times 10^{-3}$"}
        yticks_duration = np.array([0, 2, 4]) * 1e-3
        yticklabels_duration = {0: r"$0$", 2e-3: r"$2 \times 10^{-3}$", 4e-3: r"$4\times 10^{-3}$"}
        
        self._axis_ticks_and_labels(ax_time_between[0], x_ticks=x_ticks_time, x_labels=x_ticklabels_time, x_dtype="int",)
                                    # y_ticks=yticks_time_between, y_labels=yticklabels_time_between)    
        self._axis_ticks_and_labels(ax_duration[0], x_ticks=x_ticks_time, x_labels=x_ticklabels_time, x_dtype="int",)
                                    # y_ticks=yticks_duration, y_labels=yticklabels_duration)    
        xticks_NBER = [0, 3.5, 7, 10.5, 14]
        xticklabels_NBER = xticks_NBER[::2]
        yticks_NBER = [0, 7.5, 15]
        yticklabels_NBER = [0, 15]
        self._axis_ticks_and_labels(ax_time_between[1], x_ticks=xticks_NBER, x_labels=xticklabels_NBER, x_dtype="int",
                                    y_ticks=yticks_NBER, y_labels=yticklabels_NBER)
        self._axis_ticks_and_labels(ax_duration[1], x_ticks=xticks_NBER, x_labels=xticklabels_NBER, x_dtype="int",
                                    y_ticks=yticks_NBER, y_labels=yticklabels_NBER)

        # Force scientific notation on the y-axis, always showing ×10^k
        ax_time_between[0].ticklabel_format(axis='y',
                            style='scientific',
                            scilimits=(0,0),      # always use science notation
                            useMathText=True)     # render with LaTeX math
        ax_duration[0].ticklabel_format(axis='y',
                            style='scientific',
                            scilimits=(0,0),      # always use science notation
                            useMathText=True)     # render with LaTeX math


        # Company size (plot data on top)
        size_min = 0.9  # Slightly below 1 which is the true min
        size_max = 1.1*np.max((np.max(w), np.max(labels_size_data)))
        xlim_size = (size_min, size_max)
        if ylim_size is None: ylim_size = (1e-6, 2e1)
        if Nbins_size is None: Nbins_size = int(np.sqrt(np.size(w)))
        bins_size = 10 ** np.linspace(np.log10(1e-1), np.log10(np.max(w) * 10), Nbins_size)  # Log x cuts off large values if max range value is not increased
        counts_size, edges_size = np.histogram(w, bins=bins_size, density=True)
        
        ax_size[0].set(xlabel="Company size", ylabel="PMF", yscale="log", xlim=xlim_size, ylim=ylim_size, xscale="log")
        ax_size[0].stairs(counts_size, edges_size, color=self.colours["workers"], alpha=1, label="Model")
        ax_size[0].plot(labels_size_data, counts_size_data, "o", color=self.colours["workers"], mec="black", label="Data")  # mec = marker edgecolor
        ax_size[0].grid()
        ax_size[0].legend(loc="upper right", frameon=False)
        # Ticks
        ax_size[0].yaxis.set_major_locator(LogLocator(numticks=3))
        
        # Company size data
        ax_size[1].plot(labels_size_data, counts_size_data, "o", color=self.colours["workers"], mec="black")  # mec = marker edgecolor
        ax_size[1].set(yscale="log", xlabel="Company size", xscale="log", ylabel="PDF",  xlim=xlim_size, ylim=ylim_size)
        ax_size[1].grid()
        ax_size[1].yaxis.set_major_locator(LogLocator(numticks=3))
        
        # Inflation
        points_to_include = len(time_for_inflation)
        mu_smooth_inflation = mu_smooth_inflation[:points_to_include]  # Do not show hundred of thousands of points
        ax_inflation[0].plot(time_for_inflation, mu_smooth_inflation, "-", color=self.colours["mu"], lw=1)
        ax_inflation[1].plot(PCE_inflation, "-", color=self.colours["mu"])
        
        # Plot the recessions in a shaded gray
        for start, end in zip(peaks_model, troughs_model):
            ax_inflation[0].axvspan(start, end, color="gray", alpha=0.3)
        for start, end in zip(peaks_NBER, troughs_NBER):
            ax_inflation[1].axvspan(start, end, color="gray", alpha=0.3)
        xlim_inflation_data = (PCE_inflation.index[0], PCE_inflation.index[-1])
        # Axis setup
        if inflation_change_type == "log":
            ylabel_change = "Log Change"
            ylabel_change_data = ylabel_change
        else:
            ylabel_change = r" $\Delta \bar{s}_\text{real} / \bar{s}_\text{real}$"
            ylabel_change_data = "Inflation"
        
        inflation_min = 1.02 * np.min((mu_smooth_inflation.min(), PCE_inflation.min().item()))
        inflation_max = 1.02 * np.max((mu_smooth_inflation.max(), PCE_inflation.max().item()))
        ylim_inflation = (inflation_min, inflation_max)
        ax_inflation[0].set(ylabel=ylabel_change, ylim=ylim_inflation, xlabel="Time [a.u.]")
        ax_inflation[1].set(ylabel=ylabel_change_data, xlabel="Date", xlim=xlim_inflation_data)# ylim=ylim_inflation)
        # Grid
        ax_inflation[0].grid()
        ax_inflation[1].grid()
        # Set ticks for model
        xticks_inflation = np.linspace(time_for_inflation[0], time_for_inflation[-1]+1, 5)  # Time for inflation is one smaller due to diff, but that makes ticks ugly
        xticklabels_inflation = xticks_inflation[::2]
        yticks_inflation = [-1,  0, 1]

        yticklabels_inflation = {-1: "-1%", 0: "0%", 1: "1%"}
        if inflation_change_type == "log":
            yticklabels_inflation = yticks_inflation[::2]
        self._axis_ticks_and_labels(ax_inflation[0], x_ticks=xticks_inflation, x_labels=xticklabels_inflation, x_dtype="int",
                                    y_ticks=yticks_inflation, y_labels=yticklabels_inflation)
        # Extract start and end year
        start_year = PCE_inflation.index.min().year
        end_year = PCE_inflation.index.max().year
        middle_year = int((start_year + end_year) / 2)

        # Create datetime tick positions
        tick_positions = pd.to_datetime([f"{start_year}-01-01", f"{middle_year}-01-01", f"{end_year}-01-01"])
        tick_labels = [str(start_year), str(middle_year), str(end_year)]
        # Set ticks for data
        ax_inflation[1].set_xticks(tick_positions)
        ax_inflation[1].set_xticklabels(tick_labels) 
        self._axis_ticks_and_labels(ax_inflation[1], y_ticks=yticks_inflation, y_labels=yticklabels_inflation)
        
        # Add subplot labels
        for i, axis in enumerate(ax_arr.flatten()):
            self._subplot_label(axis, i, location=(0.01, 0.9))
        
        # Text, save, show
        self._text_save_show(fig, ax_arr[0, 0], f"economic_results_alpha{self.prob_expo}", xtext=0.05, ytext=0.85, fontsize=1)
        

    def plot_peak_hyperparameter_comparison(self, hyperpar_picky: dict, hyperpar_relaxed: dict, bins_picky: int, bins_relaxed: int, time_values_to_include=7000) -> None:
        """Four plots. First column show the peaks and the second column shows the corresponding distribution of recession durations. 
        First row is the picky hyperparameter and the second the relaxed selection.
        """
        # Get mu and time, then limit to the chosen time_interval
        self._get_data(self.group_name)
        mu, time = self._skip_values(self.mu, self.time_values)
        mu_smooth = uniform_filter1d(mu, size=hyperpar_picky["window_size"]) / self.W
        mu_smooth = mu_smooth[: time_values_to_include]
        time = time[: time_values_to_include]
        
        # Get the peaks and durations
        _, duration_picky, troughs_picky, peaks_picky = self._recession_time_between_and_duration(return_peaks=True, **hyperpar_picky)
        _, duration_relaxed, troughs_relaxed, peaks_relaxed = self._recession_time_between_and_duration(return_peaks=True, **hyperpar_relaxed)
        
        # Pick only the peaks and troughs inside the allowed time interval
        def _prep_extrema(extrema_data):
            """Make sure has the correct number of points (i.e. up to time_values_to_include), then shift by skip_values to get to the same space as mu

            Args:
                extrema_data (_type_): _description_

            Returns:
                _type_: _description_
            """
            idx = extrema_data[extrema_data < time_values_to_include]
            t = idx + self.skip_values
            return t, idx
        
        troughs_picky, troughs_picky_idx = _prep_extrema(troughs_picky)
        peaks_picky, peaks_picky_idx = _prep_extrema(peaks_picky)
        troughs_relaxed, troughs_relaxed_idx = _prep_extrema(troughs_relaxed)
        peaks_relaxed, peaks_relaxed_idx = _prep_extrema(peaks_relaxed)
                
        # Bin the duration data
        fig, ax_arr = plt.subplots(figsize=(10, 5), ncols=2, nrows=2)
        ax_peak_picky = ax_arr[0, 0]
        ax_peak_relaxed = ax_arr[1, 0]
        ax_dur_picky = ax_arr[0, 1]
        ax_dur_relaxed = ax_arr[1, 1]
        
        # The plots
        ax_peak_picky.plot(time, mu_smooth, c="black")
        ax_peak_picky.plot(troughs_picky, mu_smooth[troughs_picky_idx], "<", c="red")
        ax_peak_picky.plot(peaks_picky, mu_smooth[peaks_picky_idx], "^", c="green")
        
        ax_peak_relaxed.plot(time, mu_smooth, c="black")
        ax_peak_relaxed.plot(troughs_relaxed, mu_smooth[troughs_relaxed_idx], "<", c="red")
        ax_peak_relaxed.plot(peaks_relaxed, mu_smooth[peaks_relaxed_idx], "^", c="green")
        
        ax_dur_picky.hist(duration_picky, bins=bins_picky, color=self.colours["time"])
        ax_dur_relaxed.hist(duration_relaxed, bins=bins_relaxed, color=self.colours["time"])
        
        # # Legend
        # ax_peak_picky.legend(frameon=False, loc="upper right")
        # ax_peak_relaxed.legend(frameon=False, loc="upper right")
        
        # Axis setup
        ax_peak_picky.set(ylabel=r"Smoothed $P$ [a.u.]")
        ax_peak_relaxed.set(ylabel=r"Smoothed $P$ [a.u.]", xlabel="Time [a.u.]")
        ax_dur_picky.set(ylabel="Counts")
        ax_dur_relaxed.set(ylabel="Counts", xlabel="Recession duration [a.u.]")
        ax_peak_picky.grid()
        ax_peak_relaxed.grid()
        ax_dur_picky.grid()
        ax_dur_relaxed.grid()
        
        # Text, save, show
        self._text_save_show(fig, ax_arr[0, 0], f"peak_hyperpar_comparison", xtext=0.05, ytext=0.85, fontsize=1)


    def plot_inflation_comparison(self, peak_kwargs, change_type="log", window_size=10, window_size_inflation=5, annualized=True, same_ylim=False):
        # Get data
        _, mu_inflation = self._get_inflation(change_type=change_type, window_size=window_size)
        mu_smooth_inflation = uniform_filter1d(mu_inflation, size=window_size_inflation)
        time, mu, w_paid = self._skip_values(self.time_values, self.mu, self.w_paid) 
        time_for_diff = time[:-1]
        PCE, PCE_inflation = self._load_inflation_data(source="PCE", change_type=change_type, annualized=annualized)
        
        # Calculate recession peaks and troughs
        time_between, duration, troughs, peaks = self._recession_time_between_and_duration(**peak_kwargs, return_peaks=True)
        df_recession_data = self._load_peak_trough_data()
        peak_data = df_recession_data["peak"]
        trough_data = df_recession_data["trough"]

        # Update peaks to include skip_values
        for val_arr in (troughs, peaks):
            val_arr += self.skip_values
                
        # Create figure and unpack axis
        fig, ax_arr = plt.subplots(figsize=(10, 5), ncols=2, nrows=1)
        ax_mu_change, ax_PCE_change = ax_arr

        # Plots
        ax_mu_change.plot(time_for_diff, mu_smooth_inflation, "-", color=self.colours["mu"], lw=1)
        ax_PCE_change.plot(PCE_inflation, "-", color=self.colours["mu"])
        
        # Color recessions
        for start, end in zip(peaks, troughs):
            ax_mu_change.axvspan(start, end, color="gray", alpha=0.3)
        
        for start, end in zip(peak_data, trough_data):
            ax_PCE_change.axvspan(start, end, color="gray", alpha=0.3)
        
        # Axis setup
        ylabel_model = r"$\Delta \bar{s}_\text{real} / \bar{s}_\text{real}(t-1)$"
        ylabel_data = "Monthly Inflation PCEPI"
        
        inflation_min = 1.02 * np.min((mu_smooth_inflation.min(), PCE_inflation.min().item()))
        inflation_max = 1.02 * np.max((mu_smooth_inflation.max(), PCE_inflation.max().item()))
        if same_ylim:
            ylim_inflation = (inflation_min, inflation_max)
        else:
            ylim_inflation = None
        xlim_PCE = (PCE_inflation.index[0], PCE_inflation.index[-1])
        ax_mu_change.set(ylabel=ylabel_model, ylim=ylim_inflation, xlabel="Time [a.u.]")
        ax_PCE_change.set(ylabel=ylabel_data, xlabel="Date", ylim=ylim_inflation, xlim=xlim_PCE)
        # Grid
        for axis in ax_arr.flatten():
            axis.grid()
        # Ticks
        self._quick_axis_ticks(ax_mu_change, which="both", Nbins=3)
        self._quick_axis_ticks(ax_PCE_change, which="y", Nbins=3, integer=False)
            
        # Text, save, show
        self._text_save_show(fig, ax_mu_change, f"inflation_comparison_{change_type}", xtext=0.05, ytext=0.85, fontsize=1)
        plt.close()


    def plot_CDF_parameter(self, gname_list_alpha, gname_list_smin, gname_list_m, gname_list_ds, 
                           label_list_alpha, label_list_smin, label_list_m, label_list_ds,
                           xlim=None, xscale="log"):
        
        # Create figure and inpack axis
        fig, ax_arr = plt.subplots(ncols=2, nrows=2, sharey=True, sharex=True, figsize=(10, 10))
        ax_alpha = ax_arr[0, 0]
        ax_smin = ax_arr[1, 0]
        ax_m = ax_arr[0, 1]
        ax_ds = ax_arr[1, 1]

        # Tick setup
        y_ticks = [0, 0.25, 0.5, 0.75, 1]
        y_ticklabels = y_ticks[::2]
        for ax in ax_arr.flatten():
            self._axis_ticks_and_labels(ax, y_ticks=y_ticks, y_labels=y_ticklabels)

        # Define plotting functions
        def _plot_single_cdf(axis, x, label):
            # Calculate and plot CDF
            x_sort = np.sort(x)
            n = len(x_sort)
            cdf = np.arange(1, n + 1) / n
            axis.step(x_sort, cdf, where="post", label=label)
            
        def _plot_cdf_parameter(axis, index, gname_list, label_list):
            # Plot the CDF's
            for gname, label in zip(gname_list, label_list):
                # Get data                
                lifespan = self._get_lifespan(gname)
                _plot_single_cdf(axis, lifespan, label)
            
            # Figure setup
            axis.grid()
            axis.legend(frameon=False, loc="lower right", fontsize=18)
            axis.set(xlim=xlim, xscale=xscale)
            self._subplot_label(axis, index, fontsize=16)
        
        # Run the functions
        _plot_cdf_parameter(ax_alpha, 0, gname_list_alpha, label_list_alpha)
        _plot_cdf_parameter(ax_m, 1, gname_list_m, label_list_m)
        _plot_cdf_parameter(ax_smin, 2, gname_list_smin, label_list_smin)
        _plot_cdf_parameter(ax_ds, 3, gname_list_ds, label_list_ds)
        
        # Common axis setup
        ax_alpha.set(ylabel="ECDF")
        ax_smin.set(xlabel="Company Lifespan [a.u.]", ylabel="ECDF")
        ax_ds.set(xlabel="Company Lifespan [a.u.]")
        
        # Text, save, show
        self._text_save_show(fig, ax_alpha, f"lifespan_CDF", xtext=0.05, ytext=0.85, fontsize=1)
    
    
    def plot_N_W(self, gname_arr, KDE_par):
        # Get data
        self._get_data(gname_arr[0, 0])
        # For the imshow plots
        N, ratio, D, lifespan = self._load_N_W_results(gname_arr)
        # For the KDE plots
        if KDE_par["time_steps_to_include"] is None:
            time_points = self.time_values - self.skip_values
            extent_time = (self.skip_values, self.time_values)
        else:
            time_points = KDE_par["time_steps_to_include"]
            extent_time = (self.skip_values, self.skip_values+time_points)
        gname_endpoints = [gname_arr[0, -1], gname_arr[-1, -1], gname_arr[0, 0], gname_arr[-1, 0]]
        KDE_prob_arr = np.zeros((len(gname_endpoints), KDE_par["eval_points"], time_points))

        for i, gname in enumerate(gname_endpoints):
            s_eval, KDE = self._load_KDE(gname, KDE_par)
            KDE_prob_arr[i, :, :] = KDE
        
        # Create figure and subplots
        fig = plt.figure(figsize=(10, 10), constrained_layout=True)
        gs = GridSpec(nrows=4, ncols=2, figure=fig, )
        ax_D = fig.add_subplot(gs[0, :])
        ax_life = fig.add_subplot(gs[1, :])
        ax01 = fig.add_subplot(gs[2, 0])
        ax11 = fig.add_subplot(gs[2, 1])
        ax00 = fig.add_subplot(gs[3, 0])
        ax10 = fig.add_subplot(gs[3, 1])
        ax_list = [ax01, ax11, ax00, ax10]
        
        #  -- N_RATIO PLOTS --
        im = ax_D.imshow(D, aspect="auto", origin="lower", cmap="YlOrRd")
        im_life = ax_life.imshow(lifespan,  aspect="auto", origin="lower", cmap="Blues")
        
        # Axis setup
        ax_D.set(ylabel=r"$W/N$")
        ax_life.set(xlabel=r"$N$", ylabel=r"$W/N$")
        
        # Colorbar
        cbar = fig.colorbar(im, ax=ax_D, pad=0.01)
        cbar.set_label(label=r"$\bar{D} / N$", fontsize=16)
        cbar_life = fig.colorbar(im_life, ax=ax_life, pad=0.01)
        cbar_life.set_label(label="Median Lifetime", fontsize=16)
        cbar.locator = MaxNLocator(nbins=3)  # Request ~4 ticks
        cbar.update_ticks()
        cbar_life.locator = MaxNLocator(nbins=3)  # Request ~4 ticks
        cbar_life.update_ticks()
        
        # Ticks
        # Set ticks in the center of each block
        def _ticker(axis, Z, data_type):
            axis.set_xticks(np.arange(len(N)))
            axis.set_yticks(np.arange(len(ratio)))
            # Set tick labels to actual values
            axis.set_xticklabels(N)
            axis.set_yticklabels(ratio)
            # Show tick grid lines aligned with blocks
            axis.set_xticks(np.arange(len(N)+1)-0.5, minor=True)
            axis.set_yticks(np.arange(len(ratio)+1)-0.5, minor=True)
            axis.grid(which="minor", color="grey", linestyle='-', linewidth=0.5)
            axis.tick_params(which="minor", bottom=False, left=False)
            # Print the values of each block
            for i in range(len(ratio)):
                for j in range(len(N)):
                    if data_type == "float":
                        axis.text(j, i, f"{Z[i, j]:.2f}", ha="center", va="center", color="black")
                    else:
                        axis.text(j, i, f"{Z[i, j]:.0f}", ha="center", va="center", color="black")
        
        _ticker(ax_D, D, "float")
        _ticker(ax_life, lifespan, "int")
        # Subplot labels
        self._subplot_label(ax_D, index=0)
        self._subplot_label(ax_life, index=1)
        
        # -- KDE PLOTS --
        KDE_min = np.min(KDE_prob_arr)
        KDE_max = np.max(KDE_prob_arr)
        suffix_list = [fr"$: N={N[0]}, W/N={ratio[-1]}$",
                       fr"$: N={N[-1]}, W/N={ratio[-1]}$",
                       fr"$: N={N[0]}, W/N={ratio[0]}$",
                       fr"$: N={N[-1]}, W/N={ratio[0]}$",]
        x_ticks = np.linspace(extent_time[0], extent_time[1], 5)
        x_ticklabels = x_ticks[::2]
        y_ticks = np.linspace(0, KDE_par["data_lim"][1], 3)
        y_ticklabels = np.round(y_ticks[::2], 2)
        
        for i, axis in enumerate(ax_list):
            im_KDE = axis.imshow(KDE_prob_arr[i], aspect="auto", origin="lower",
                        extent=[extent_time[0], extent_time[1], np.min(s_eval), np.max(s_eval)],
                        cmap="hot", vmin=KDE_min, vmax=KDE_max)
            
            self._subplot_label(axis, index=2+i, suffix=suffix_list[i], color="white", outline_color="darkred")
            self._axis_ticks_and_labels(axis, x_ticks, y_ticks, x_ticklabels, y_ticklabels, x_dtype="int")
        # Remove inner ticklabels
        ax01.set_xticklabels([])
        ax11.set_xticklabels([])
        ax10.set_yticklabels([])
        ax11.set_yticklabels([])
        # Axis setup
        ax00.set(xlabel="Time [a.u.]", ylabel="Wage [a.u.]")
        ax10.set(xlabel="Time [a.u.]")
        ax01.set(ylabel="Wage [a.u.]")
        # Add cbar
        cbar_ax = fig.add_axes([0.908, 0.035, 0.01, 0.44])  # [left, bottom, width, height]
        fig.colorbar(im_KDE, cax=cbar_ax, label="Wage Frequency", ticks=[])
        # Text, save, show
        self._text_save_show(fig, ax_life, f"N_W", xtext=0.05, ytext=0.85, fontsize=1)


    def _trimmed_spread_over_mean(self, wages, lower_pct=10, upper_pct=95, hard_lower_bound=None):
        """
        Calculate the spread over mean of a wage distribution,
        excluding the tails (e.g. top and bottom 5%).

        Parameters:
            wages (np.ndarray or pd.Series): Array of wage values
            lower_pct (float): Lower percentile cutoff (e.g. 5 for bottom 5%)
            upper_pct (float): Upper percentile cutoff (e.g. 95 for top 5%)

        Returns:
            float: Spread over mean (std / mean) of the trimmed data
        """
        if hard_lower_bound is not None:
            wages = wages[wages>hard_lower_bound]
        lower_bound = np.percentile(wages, lower_pct)
        upper_bound = np.percentile(wages, upper_pct)
        trimmed = wages[(wages >= lower_bound) & (wages <= upper_bound)]
        return np.std(trimmed) / np.mean(trimmed)


    def _tau_table(self, lower_pct, upper_pct, tau_values, N_values, hard_lower_bound):
        self._get_data(self.group_name)
        SoM_arr = np.zeros((len(tau_values), len(N_values)))
        for i, tau in enumerate(tau_values):
            for j, N in enumerate(N_values):
                if N == 10_000:
                    hard_lower_bound_val = hard_lower_bound
                else:
                    hard_lower_bound_val = None
                # Get data and calculate SoM
                gname = self.new_gname(N=N, number_of_transactions_per_step=tau)
                self._get_data(gname)
                mu, s = self._skip_values(self.mu, self.s)
                # average_wage = mu / self.W
                # SoM = np.std(average_wage) / np.mean(average_wage)
                SoM = self._trimmed_spread_over_mean(s, lower_pct, upper_pct, hard_lower_bound_val)  # Get rid of outlier values.
                # Store
                SoM_arr[i, j] = SoM
        
        return SoM_arr


    def format_tau_table_latex(self, tau_values, N_values, alpha_values, lower_pct, upper_pct, hard_lower_bound):
        SoM_arr = self._tau_table(lower_pct, upper_pct, tau_values, N_values, hard_lower_bound)
        headers = ["$\\tau \\backslash N$"] + [f"{N}" for N in N_values]
        table_data = []

        for i, tau in enumerate(tau_values):
            row = [f"{tau}"] + [f"{SoM_arr[i, j]:.2f}" for j in range(len(N_values))]
            table_data.append(row)

        latex_table = tabulate(table_data, headers, tablefmt="latex")
        return latex_table



    def trimmed_mean_over_N_vectorized(self, s, lower_pct=5, upper_pct=95, hard_lower_bound=0.04):
        """
        Compute the trimmed mean of s over N for each time step (axis=0),
        applying a hard lower bound and trimming by percentiles, all vectorized.
        
        s: np.ndarray, shape (N, T)
        """
        # 1) Mask out everything below the hard lower bound
        s_masked = s.copy()
        if hard_lower_bound is not None:
            s_masked[s_masked < hard_lower_bound] = np.nan

        # 2) Compute the per-column (per-time) percentiles, ignoring NaNs
        lower = np.nanpercentile(s_masked, lower_pct, axis=0)  # shape (T,)
        upper = np.nanpercentile(s_masked, upper_pct, axis=0)  # shape (T,)

        # 3) Mask out everything outside [lower, upper]
        #    We need to broadcast lower/upper across rows:
        too_low  = s_masked < lower[np.newaxis, :]
        too_high = s_masked > upper[np.newaxis, :]
        s_masked[too_low | too_high] = np.nan

        # 4) Finally, take the mean over N (rows), ignoring NaNs
        #    This gives an array of length T
        return np.nanmean(s_masked, axis=0)


    def plot_tau(self, lower_pct, upper_pct, hard_lower_bound=None, time_values_to_show=None, 
                 tau_values=[1, 2, 4, 100], N=1000, bandwidth=0.005, eval_points=250, kernel="gaussian",
                 s_lim=(0, 0.2), show_mean=True):
        # Create figure
        fig, ax_list = plt.subplots(nrows=len(tau_values), ncols=1, sharex=True, sharey=True)
        
        # Get xlim
        self._get_data(self.group_name)        
        if time_values_to_show is None:
            xlim = (self.skip_values, self.time_steps)
        else:
            xlim = (self.skip_values, self.skip_values+time_values_to_show)        
        x_ticks = np.linspace(xlim[0], xlim[-1], 5).astype(np.int32)
        x_ticklabels = x_ticks[::2]
        
        # Load SoM values
        SoM_arr = self._tau_table(tau_values=tau_values, N_values=[N], lower_pct=lower_pct, upper_pct=upper_pct, hard_lower_bound=hard_lower_bound).flatten()
        # Create plotter function
        def _plotter(idx):
            # Get axis and gname
            axis = ax_list[idx]
            tau = tau_values[idx]            
            gname = self.new_gname(number_of_transactions_per_step=tau, N=N)
            # Get KDE
            s_eval, KDE = self.running_KDE("salary", bandwidth, eval_points, kernel, s_lim, gname)
            # Get wage
            s, time, mu = self._skip_values(self.s, self.time_values, self.mu)
            if N == 10_000:
                hard_lower_bound_val = hard_lower_bound
            else:
                hard_lower_bound_val = None
            s_trimmed = self.trimmed_mean_over_N_vectorized(s, lower_pct, upper_pct, hard_lower_bound_val)
            # s_trimmed = mu / self.W
            # Get spread / mean
            SoM = SoM_arr[idx]
            # SoM = np.std(s) / np.mean(s)  # OBS Maybe has to cut off the lowest 5% as we only want the mean of the cycle values
            # Plot KDE
            axis.imshow(KDE, aspect="auto", origin="lower", extent=[self.skip_values, self.time_steps, s_lim[0], s_lim[-1]], cmap="hot")
            if show_mean: axis.plot(time, s_trimmed, c="magenta", lw=0.55)
            axis.set(ylabel="Wage [$]", xlim=xlim)
            # Label
            self._subplot_label(axis, idx, suffix=fr"), $\tau={tau}$, SoM = {SoM:.2f}", color="white", outline_color="black")
            # y ticks
            self._quick_axis_ticks(axis, which="y", Nbins=3, integer=False)        
            self._axis_ticks_and_labels(axis, x_ticks=x_ticks, x_labels=x_ticklabels, x_dtype="int")
        
        for i in tqdm(range(len(tau_values))):
            _plotter(i)
        
        ax_list[-1].set(xlabel="Time [a.u.]")
        # Text, save, show
        self._text_save_show(fig, ax_list[0], f"tau", xtext=0.05, ytext=0.85, fontsize=1)        



    def plot_inject_money(self, inject_money_time, bandwidth, s_lim, eval_points=250, kernel="gaussian", show_mean=True):
        # Load data by getting the gname and change one to have inject_money_time=0 and one to the given variable
        self._get_data(self.group_name)
        
        gname_inj0 = self.new_gname(inject_money_time=0)
        gname_inj = self.new_gname(inject_money_time=inject_money_time)
        
        s_eval_inj0, KDE_prob_inj0 = self.running_KDE("salary", bandwidth, eval_points, kernel, s_lim, gname=gname_inj0)  # KDE probabilities        
        time, mu_inj0 = self._skip_values(self.time_values, self.mu)
        
        s_eval_inj, KDE_prob_inj = self.running_KDE("salary", bandwidth, eval_points, kernel, s_lim, gname=gname_inj)  # KDE probabilities        
        mu_inj = self._skip_values(self.mu)
        
        fig, (ax, ax_inj) = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
        
        # The no injection i.e. at 0
        im = ax.imshow(KDE_prob_inj0, aspect="auto", origin="lower", extent=[self.skip_values, self.time_steps, np.min(s_eval_inj0), np.max(s_eval_inj0)], cmap="hot")
        if show_mean: ax.plot(time, mu_inj0/self.W, c="magenta", lw=0.6)
        ax.set_ylabel("Wage [$]")
    
        # Injection
        im_inj = ax_inj.imshow(KDE_prob_inj, aspect="auto", origin="lower", extent=[self.skip_values, self.time_steps, np.min(s_eval_inj), np.max(s_eval_inj)], cmap="hot")
        if show_mean: ax_inj.plot(time, mu_inj/self.W, c="magenta", lw=0.6)
        ax_inj.set(ylabel="Wage [$]", xlabel="Time [a.u.]")

        # Axis ticks
        x_max = self.time_steps - self.skip_values
        x_min = self.skip_values
        x_ticks = np.linspace(x_min, x_max, 5)
        self._quick_axis_ticks(ax, which="y", Nbins=3, integer=False,)
        self._quick_axis_ticks(ax, which="x", Nbins=3, integer=True,)

        # Subplot labels
        self._subplot_label(ax, 0, color="white", suffix="): Standard")
        self._subplot_label(ax_inj, 1, color="white", suffix="): Injecting money")

        # Text, save, show
        self._text_save_show(fig, ax, f"injection", xtext=0.05, ytext=0.85, fontsize=1)