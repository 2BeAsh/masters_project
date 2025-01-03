import general_functions 
import matplotlib.pyplot as plt
import numpy as np
import h5py
from pathlib import Path
from run import file_path, group_name, dir_path_image


class PlotMaster(general_functions.PlotMethods):
    def __init__(self, data_group_name, show_plots=True, add_parameter_text_to_plot=True, save_figure=True):
        super().__init__(data_group_name)
        # Allow for the option of specifying a group name. Otherwise the existing group name is used.
        self.group_name = data_group_name
        self.show_plots = show_plots
        self.save_figure = save_figure
        self.add_parameter_text_to_plot = add_parameter_text_to_plot
        self.dir_path_image = dir_path_image
        self.loaded_groups = {}
        
        self.skip_values = 2500
        
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
                "first_company_went_bankrupt": np.array(group.get("first_company_went_bankrupt", None)),
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
        self.first_company_went_bankrupt = data["first_company_went_bankrupt"]
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
        ax0.plot(time_values, mean_salary, label="Mean salary", c=c0, alpha=1)
        ax0.plot(time_values, median_salary, label="Median salary", c="black", alpha=0.5, ls="dotted")
        
        ax0.set(xlim=self.xlim, ylabel="Log Price", yscale="log", title="Mean salary and bankruptcies")
        ax0.set_ylabel("Log Price", color=c0)
        ax0.tick_params(axis='y', labelcolor=c0)
        ax0.grid()
        self._add_legend(ax0, ncols=2, x=0.5, y=0.9)

        ax0_twin = ax0.twinx()
        ax0_twin.plot(time_values, fraction_bankrupt, color=c1, label="Fraction bankrupt", alpha=0.6)
        ax0_twin.set_ylabel("Fraction bankrupt", color=c1)
        ax0_twin.tick_params(axis='y', labelcolor=c1)
        
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
        ax_s.plot(time_values, s.T)
        ax_s.set(title=f"Salary and debt of first {N_plot} companies", yscale="log")
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
        time_bankrupt = time_values[self.first_company_went_bankrupt[self.skip_values:] == 1]
        y_bankrupt = np.zeros(len(time_bankrupt))
        ax_d.scatter(time_bankrupt, y_bankrupt, c="black", marker="x", s=7)

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
        