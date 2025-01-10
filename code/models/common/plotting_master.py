import general_functions 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import functools
import numpy as np
import h5py
from pathlib import Path
from run import file_path, group_name, dir_path_image


class PlotMaster(general_functions.PlotMethods):
    def __init__(self, data_group_name, skip_values=0, show_plots=True, add_parameter_text_to_plot=True, save_figure=True):
        super().__init__(data_group_name)
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
        }
    
        
    def _load_data_group(self, gname):
        with h5py.File(file_path, "r") as f:
            group = f[gname]
            data = {
                "w": np.array(group.get("w", None)),
                "d": np.array(group.get("d", None)),
                "s": np.array(group.get("s", None)),
                "r": np.array(group.get("r", None)),
                "went_bankrupt": np.array(group.get("went_bankrupt", None)),
                "went_bankrupt_idx": np.array(group.get("went_bankrupt_idx", None)),
                "mu": np.array(group.get("mu", None)),
                "mutations": np.array(group.get("mutations", None)),
                "peak_idx": np.array(group.get("peak_idx", None)),
                "repeated_m_runs": np.array(group.get("repeated_m_runs", None)),
                "N": np.array(group.attrs.get("N", None)),
                "time_steps": group.attrs.get("time_steps", None),
                "W": np.array(group.attrs.get("W", None)),
                "ds": np.array(group.attrs.get("ds", None)),
                "rf": np.array(group.attrs.get("rf", None)),
                "m": np.array(group.attrs.get("m", None)),
                "prob_expo": np.array(group.attrs.get("prob_expo", None)),
                "m_repeated": np.array(group.attrs.get("m_repeated", None)),
            }
            self.loaded_groups[gname] = data
    
    
    def _get_data(self, gname):
        """Load data from gname if it has not already been loaded."""
        if gname not in self.loaded_groups:
            self._load_data_group(gname)
        # Set the attributes from the loaded data
        data = self.loaded_groups[gname]
        self.w = data["w"]
        self.d = data["d"]
        self.s = data["s"]
        self.r = data["r"]
        self.went_bankrupt = data["went_bankrupt"]
        self.went_bankrupt_idx = data["went_bankrupt_idx"]
        self.mu = data["mu"]
        self.mutations = data["mutations"]
        self.N = data["N"]
        self.time_steps = data["time_steps"]
        self.W = data["W"]
        self.ds = data["ds"]
        self.rf = data["rf"]
        self.m = data["m"]
        self.prob_expo = data["prob_expo"]
        self.peak_idx = data["peak_idx"]
        self.salary_repeated_m_runs = data["repeated_m_runs"]
        self.m_repeated = data["m_repeated"]
        
        if (self.s != None).all():
            self.xlim = (self.skip_values, self.time_steps)
            
            if self.time_steps <= self.skip_values:
                self.skip_values = 0
                print(f"Skip values {self.skip_values} is larger than the time steps {self.time_steps}. Set skip values to 0.")
    
    
    def plot_salary(self):
        """Plot the mean salary and fraction who went bankrupt on twinx. Plot the spread (std/mean) on a subplot below it."""
        self._get_data(self.group_name)
        mean_salary = self.s.mean(axis=0)[self.skip_values:]
        median_salary = np.median(self.s, axis=0)[self.skip_values:]
        fraction_bankrupt = (self.went_bankrupt[self.skip_values:] / self.N)
        spread = (self.s.std(axis=0)[self.skip_values:] / mean_salary)
        time_values = np.arange(self.skip_values, self.time_steps)
        
        # Create figure
        fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(10, 10))
        
        # ax0 - Salary and fraction who went bankrupt
        c0 = self.colours["salary"]
        c1 = self.colours["bankruptcy"]
        
        # Bankruptcy
        ax0_twin = ax0.twinx()
        ax0_twin.plot(time_values, fraction_bankrupt, color=c1, label="Fraction bankrupt", alpha=0.6)
        ax0_twin.set_ylabel("Fraction bankrupt", color=c1)
        ax0_twin.tick_params(axis='y', labelcolor=c1)

        # Mean and median salary
        ax0.plot(time_values, mean_salary, label="Mean salary", c=c0, alpha=1)
        ax0.plot(time_values, median_salary, label="Median salary", c="black", alpha=0.7, ls="dotted")
        ax0.set(xlim=self.xlim, ylabel="Log Price", yscale="log", title="Mean salary and bankruptcies")
        ax0.set_ylabel("Log Price", color=c0)
        ax0.tick_params(axis='y', labelcolor=c0)
        ax0.grid()
        self._add_legend(ax0, ncols=3, x=0.5, y=0.9)
        
        # ax1 - Spread
        ax1.plot(time_values, spread, label="Spread")
        ax1.set(xlabel="Time", xlim=self.xlim, ylabel="Spread", title="Spread (std/mean)")
        ax1.grid()
        
        # Plot the peaks as vertical lines on ax0 and ax1
        if np.any(self.peak_idx != None):
            for peak in self.peak_idx:
                ax0.axvline(x=peak, ls="--", c="grey", alpha=0.7)
                ax1.axvline(x=peak, ls="--", c="grey", alpha=0.7)
        
        self._text_save_show(fig, ax0, "salary", xtext=0.05, ytext=0.75, fontsize=6)
        
        
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
        fig, (ax_s, ax_d, ax_w) = plt.subplots(nrows=3)
        
        # ax_s - salary
        ylim = (np.median(s) / 4, np.max(s)*1.01)
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
        mean_debt = self.d.mean(axis=0)[self.skip_values:]
        median_debt = np.median(self.d, axis=0)[self.skip_values:]
        mean_salary = self.s.mean(axis=0)[self.skip_values:]
        fraction_bankrupt = (self.went_bankrupt[self.skip_values:] / self.N)
        time_values = np.arange(self.skip_values, self.time_steps)
        d_final = self.d[:, -1]
        
        # Create figure
        fig, (ax, ax1, ax2) = plt.subplots(nrows=3)
        c0 = self.colours["debt"]
        c1 = self.colours["bankruptcy"]
        
        ax.plot(time_values, mean_debt, c=c0, label="Mean debt")
        # ax.plot(time_values, median_debt, c=c0, ls="--", label="Median debt")
        ax.set(xlabel="Time", title="Mean debt and bankruptcies", yscale="symlog")
        ax.set_ylabel("Log Price", color=c0)
        ax.tick_params(axis='y', labelcolor=c0)
        ax.grid()
        self._add_legend(ax, ncols=2, x=0.7, y=0.7, fontsize=6)
        
        ax_twin = ax.twinx()
        ax_twin.plot(time_values, fraction_bankrupt, color=c1, label="Fraction bankrupt", alpha=0.6)
        ax_twin.set_ylabel("Fraction bankrupt", color=c1)
        ax_twin.tick_params(axis='y', labelcolor=c1)
        
        # ax1 - Salary and debt
        c2 = self.colours["salary"]
        ax1.plot(time_values, mean_debt, c=c0)
        ax1.set(xlabel="Time", title="Mean salary and debt", yscale="symlog")
        ax1.set_ylabel("Mean Debt", color=c0)
        ax1.tick_params(axis='y', labelcolor=c0)
        ax1.grid()
        
        ax1_twin = ax1.twinx()
        ax1_twin.plot(time_values, mean_salary, c=c2, alpha=0.7)
        ax1_twin.set_ylabel("Log mean salary", color=c2)
        ax1_twin.tick_params(axis='y', labelcolor=c2)
        ax1_twin.set_yscale("log")
        
        # ax2 - Debt distribution
        Nbins = int(np.sqrt(self.N))
        ax2.hist(d_final, bins=Nbins, color=c0)
        ax2.set(title="Debt distribution at final time step", xlabel="Debt", ylabel="Counts", yscale="log")
        ax2.grid()
        
        # Log scale hist requires only positive values
        # self._xlog_hist(d_final, fig, ax2, xlabel="Log Debt", ylabel="Counts", title="Debt distribution at final time step")
        
        self._text_save_show(fig, ax, "debt", xtext=0.05, ytext=0.85)
        
        
    def plot_collective(self):
        """Plot the mean salary and fraction bankrupt on a shared x axis, plot debt in another subplot below it, plot interest rate in a third subplot.
        """
        # Preprocess
        self._get_data(self.group_name)
        s_mean = self.s.mean(axis=0)[self.skip_values:]
        d_mean = self.d.mean(axis=0)[self.skip_values:]
        w_mean = self.w.mean(axis=0)[self.skip_values:]
        w_bankrupt = self.went_bankrupt[self.skip_values:]
        r = self.r[self.skip_values:]
        time_values = np.arange(self.skip_values, self.time_steps)
        
        fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(14, 10))
        ax_s = ax[0, 0]
        ax_d = ax[1, 0]
        ax_r = ax[0, 1]
        ax_w = ax[1, 1]
        
        # ax_s - Salary and bankrupt
        c0 = self.colours["salary"]
        c1 = self.colours["bankruptcy"]
        ax_s_twin = ax_s.twinx()
        ax_s_twin.plot(time_values, w_bankrupt / self.N, c=c1, alpha=0.7)
        ax_s_twin.set_ylabel("Fraction bankrupt", color=c1)
        ax_s_twin.tick_params(axis='y', labelcolor=c1)

        ax_s.plot(time_values, s_mean, c=c0)
        ax_s.tick_params(axis='y', labelcolor=c0)
        ax_s.set_ylabel("Log salary", color=c0)
        ax_s.set(title="Mean salary and bankruptcies", yscale="log")
        ax_s.grid()

        # ax_d - Debt
        c2 = self.colours["debt"]
        ax_d.plot(time_values, d_mean, c=c2)
        ax_d.set(title="Log Mean debt", xlabel="Time", yscale="symlog")
        ax_d.grid()

        # ax_r - Interest rate
        c3 = self.colours["interest_rate"]
        ax_r.plot(time_values, r, c=c3)
        ax_r.set(title="Interest rate", xlabel="Time")
        ax_r.grid()
        
        # ax_w - Workers
        c4 = self.colours["workers"]
        ax_w.plot(time_values, w_mean, c=c4)
        ax_w.set(title="Workers", xlabel="Time")
        ax_w.grid()

        self._text_save_show(fig, ax_s, "collective", xtext=0.05, ytext=0.85)
        
    
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
    
        
    def plot_multiple_mutation_size(self, group_name_list):
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
            m = self.m
            mean_salary_list[i] = mean_salary
            mutation_size_list[i] = m
            
        # Get min and max salary, divide by mutation size
        min_salary = np.min(mean_salary_list, axis=1) / mutation_size_list
        max_salary = np.max(mean_salary_list, axis=1) / mutation_size_list
        
        # Create figure
        fig, (ax, ax1) = plt.subplots(nrows=2)
        c_list = general_functions.list_of_colors[:len(group_name_list)]
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
        self._text_save_show(fig, ax, "multiple_mutation_size", xtext=0.05, ytext=0.75)
    
    
    def plot_repeated_mutation_size(self):
        """Plot the mean of repeated measurements of the minimum and maximum of the mean salary, together with their uncertanties.
        """
        # Calculate mean and std of min and max salary for each m
        self._get_data(self.group_name)
        if np.any(self.m_repeated == None):
            raise ValueError("No repeated m runs data found.")
        N_repeats = np.shape(self.salary_repeated_m_runs)[1]
        time_steps = np.shape(self.salary_repeated_m_runs)[2]
        
        min_arr = np.min(self.salary_repeated_m_runs, axis=2)
        mean_min_arr = np.mean(min_arr, axis=1) / self.m_repeated  # Normalize the minimum salary by the mutation magnitude
        std_mean_min_arr = np.std(min_arr, axis=1, ddof=1) / np.sqrt(N_repeats) / self.m_repeated
        
        max_arr = np.max(self.salary_repeated_m_runs, axis=2) 
        mean_max_arr = np.mean(max_arr, axis=1) / self.m_repeated
        std_mean_max_arr = np.std(max_arr, axis=1, ddof=1) / np.sqrt(N_repeats) / self.m_repeated

        fig, ax = plt.subplots()
        ax.errorbar(self.m_repeated, mean_min_arr, yerr=std_mean_min_arr, fmt="v", label=r"$\min(\bar{s}) / m $", color="k")
        ax.errorbar(self.m_repeated, mean_max_arr, yerr=std_mean_max_arr, fmt="^", label=r"$\max(\bar{s}) / m $", color="k")
        ax.set(xlabel=r"$m$", ylabel="Log Price", yscale="log", xscale="log")
        ax.set_title(f"Repeated measurements of min and max salary, N={N_repeats}, t={time_steps}", fontsize=10)
        ax.grid()
        
        self._add_legend(ax, ncols=2, y=0.9)
        self._save_fig(fig, "repeated_mutation_size")
        plt.show()
            
        
    def plot_multiple_prob_expo(self, group_name_list):
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
        # Calculate nrows and ncols
        nrows = 2
        ncols = (len(group_name_list) + nrows - 1) // nrows
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 8))
        c_list = general_functions.list_of_colors[:len(group_name_list)]
        
        for i, (mean_salary, expo) in enumerate(zip(mean_salary_list, prob_expo_list)):
            ax = axs[i//ncols, i%ncols]
            ax.plot(time_vals, mean_salary, c=c_list[i])
            ax.set_title(fr"Exponent = {int(expo)}", fontsize=8)
            ax.set(yscale="log")
            ax.grid()

            # Axis labels. Only the bottom row should have x labels, and only the left column should have y labels
            subplot_spec = ax.get_subplotspec()
            if subplot_spec.is_last_row():
                ax.set_xlabel("Time")
            if subplot_spec.is_first_col():
                ax.set_ylabel("Log Price")
        
        # save show
        self._save_fig(fig, "multiple_prob_expo")
        plt.show()
        
        
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
        s_bankrupt = self.s[self.went_bankrupt_idx == 1]
        N_bankrupt = np.sum(self.went_bankrupt_idx)
        # Skip values
        self.s = self.s[:, self.skip_values:]
        s_bankrupt = s_bankrupt[self.skip_values:]
        # Bins
        Nbins = int(np.sqrt(N_bankrupt))
        logbins = np.geomspace(np.min(self.s), np.max(self.s), Nbins)
        # Histogram
        counts, edges = np.histogram(s_bankrupt, bins=logbins)
        
        fig, (ax, ax1) = plt.subplots(nrows=2)
        ax.hist(edges[:-1], edges, weights=counts, color=self.colours["salary"])
        ax.set(ylabel="Log Counts", yscale="log", xscale="log", title="New salary of bankrupt companies")
        ax.grid()
        
        ax1.hist(np.ravel(self.s), bins=logbins, color=self.colours["salary"])
        ax1.set(xlabel="Log Salary", ylabel="Log Counts", yscale="log", xscale="log", title="All salary values")
        ax1.grid()
        
        # Text, save, show
        self._text_save_show(fig, ax, "bankrupt_new_salary", fontsize=6)
        

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
        
        