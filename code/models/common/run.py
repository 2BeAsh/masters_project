import numpy as np
from pathlib import Path
import h5py
from define_methods import MethodsWorkForce


# Local paths for saving files
path_to_file = Path(__file__)
dir_path = path_to_file.parent.parent.parent
dir_path_output = Path.joinpath(dir_path, "output")
dir_path_image = Path.joinpath(dir_path, "images", "common")
dir_path_image.mkdir(parents=True, exist_ok=True)
# File name and path
file_name = "common.h5"
file_path = dir_path_output / file_name


class RunWorkForce(MethodsWorkForce):
    def __init__(self, number_of_companies, number_of_workers, salary_increase, interest_rate_free, mutation_magnitude, prob_exponent, salary_min, update_methods, time_steps, seed):
        """Functions for running the simulation and storing the data.

        Args:
            number_of_companies (_type_): _description_
            number_of_workers (_type_): _description_
            salary_increase (_type_): _description_
            interest_rate_free (_type_): _description_
            mutation_magnitude (_type_): _description_
            prob_exponent (_type_): _description_
            update_methods (_type_): _description_
            time_steps (_type_): _description_
            seed (_type_): _description_
        """
        # Inherit from MethodsWorkForce which inherits from WorkForce
        self.worker_update_method = update_methods["worker_update"]
        self.prob_exponent = prob_exponent
        if self.worker_update_method == "unlimited":
            number_of_workers = number_of_companies
        self.rf_name = interest_rate_free
        
        super().__init__(number_of_companies, number_of_workers, salary_increase, interest_rate_free, mutation_magnitude, salary_min, update_methods, time_steps, seed)


    def store_data_in_group(self, print_info=True):
        # Run simulation to get data
        self.group_name = self._get_group_name()
        if print_info: print(f"Storing data in {self.group_name}")
        self._simulation()
                
        with h5py.File(file_path, "a") as file:
            # Try and open the file if it already exists
            try:
                group = file[self.group_name]
            except KeyError:
                group = file.create_group(self.group_name)
            
            # If any of the data already exists, delete all of them (if one is present, all of them are present)
            if "w" in group:
                del group["w"]
                del group["d"]
                del group["s"]
                
                del group["r"]
                del group["went_bankrupt"]
                del group["went_bankrupt_idx"]
                del group["mu"]
                del group["mutations"]
                
            # Store data in group
            # Company
            group.create_dataset("w", data=self.w_hist)
            group.create_dataset("d", data=self.d_hist)
            group.create_dataset("s", data=self.s_hist)
            # System
            group.create_dataset("r", data=self.r_hist)
            group.create_dataset("went_bankrupt", data=self.went_bankrupt_hist)
            group.create_dataset("went_bankrupt_idx", data=self.went_bankrupt_idx_hist)
            group.create_dataset("mu", data=self.mu_hist)
            group.create_dataset("mutations", data=self.mutations_hist)
            # Attributes
            group.attrs["N"] = self.N
            group.attrs["time_steps"] = self.time_steps
            group.attrs["W"] = self.W
            group.attrs["ds"] = self.ds
            group.attrs["rf"] = self.rf
            group.attrs["m"] = self.mutation_magnitude
            group.attrs["prob_expo"] = self.prob_exponent
            group.attrs["s_min"] = self.salary_min
            
            
    def repeated_m_runs(self, N_repeat, m_values):
        self.group_name = self._get_group_name()
        print(f"Storing repeated m runs in {self.group_name}")
        mean_salary_arr = np.zeros((len(m_values), N_repeat, self.time_steps))
        total_iterations = len(m_values) * N_repeat
        for i, m in enumerate(m_values):
            for j in range(N_repeat):
                print(f"iteration: {i*N_repeat + j+1}/{total_iterations}")
                self.mutation_magnitude = m
                # self.salary_min = m / 10
                self._simulation()
                mean_salary = np.mean(self.s_hist, axis=0)  # Over companies
                mean_salary_arr[i, j, :] = mean_salary
                
        # Save data            
        with h5py.File(file_path, "a") as file:
            try:
                data_group = file[self.group_name]
            except KeyError:
                data_group = file.create_group(self.group_name)
            if "repeated_m_runs" in data_group:
                del data_group["repeated_m_runs"]
                del data_group.attrs["m_repeated"]
            data_group.create_dataset("repeated_m_runs", data=mean_salary_arr)
            data_group.attrs["m_repeated"] = m_values
            print(f"Stored repeated m runs in {self.group_name}")
        
            
    def multiple_s_min_runs(self, s_min_list):
        """Generate data for different s_min values.

        Args:
            s_min_list (list): _description_
        """
        self.group_name = self._get_group_name()
        print(f"Storing multiple s_min runs in {self.group_name}")
        s_arr = np.zeros((len(s_min_list), self.N, self.time_steps))
        bankruptcy_arr = np.zeros((len(s_min_list), self.time_steps))
        total_iterations = len(s_min_list)
        s_min_current = self.salary_min
        for i, s_min in enumerate(s_min_list):
            print(f"iteration: {i+1}/{total_iterations}")
            self.salary_min = s_min
            self._simulation()
            s_arr[i, :, :] = self.s_hist
            bankruptcy = self.went_bankrupt_hist / self.N
            bankruptcy_arr[i, :] = bankruptcy * 1
        
        # Save data
        with h5py.File(file_path, "a") as file:
            try:
                data_group = file[self.group_name]
            except KeyError:
                data_group = file.create_group(self.group_name)
            if "s_s_min" in data_group:
                del data_group["s_s_min"]
            if "bankruptcy_s_min" in data_group:
                del data_group["bankruptcy_s_min"]
            if "s_min_list" in data_group.attrs:
                del data_group.attrs["s_min_list"]
            
            data_group.create_dataset("s_s_min", data=s_arr)
            data_group.create_dataset("bankruptcy_s_min", data=bankruptcy_arr)
            data_group.attrs["s_min_list"] = s_min_list
        
        # Reset s_min
        self.salary_min = s_min_current


    def multiple_ds_runs(self, ds_list):
        """Generate data for different ds values.

        Args:
            ds_list (list): _description_
        """
        self.group_name = self._get_group_name()
        self._simulation()
        print(f"Storing multiple ds runs in {self.group_name}")
        s_arr = np.zeros((len(ds_list), self.N, self.time_steps))
        bankruptcy_arr = np.zeros((len(ds_list), self.time_steps))
        total_iterations = len(ds_list)
        ds_current = self.ds
        for i, ds in enumerate(ds_list):
            print(f"iteration: {i+1}/{total_iterations}")
            self.ds = ds
            self._simulation()
            s_arr[i, :, :] = self.s_hist
            bankruptcy = self.went_bankrupt_hist / self.N
            bankruptcy_arr[i, :] = bankruptcy * 1
        
        # Save data
        with h5py.File(file_path, "a") as file:
            try:
                data_group = file[self.group_name]
            except KeyError:
                data_group = file.create_group(self.group_name)
            if "s_ds" in data_group:
                del data_group["s_ds"]
            if "bankruptcy_ds" in data_group:
                del data_group["bankruptcy_ds"]
            if "ds_list" in data_group.attrs:
                del data_group.attrs["ds_list"]
            
            data_group.create_dataset("s_ds", data=s_arr)
            data_group.create_dataset("bankruptcy_ds", data=bankruptcy_arr)
            data_group.attrs["ds_list"] = ds_list
        
        # Reset s_min
        self.ds = ds_current
            

    def generate_m_arr_data(self, m_vals, N_repeat, linear_smin: bool, store_data=True, alpha=None, N=None, W=None):
        """Store data for different m values with N_repeat repeats or if store_data=False only return group names.

        Args:
            m_vals (_type_): _description_
            N_repeat (_type_): _description_

        Returns:
            np.ndarray: Group name arrays for ds values with repeats
        """
        # Get current values such that they can later be reset to these
        current_N = self.N
        current_W = self.W
        current_seed = self.seed
        current_prob_exponent = self.prob_exponent
        current_m = self.mutation_magnitude
        current_smin = self.salary_min
        
        # Set alpha, N and W if given, otherwise they are the default values
        if alpha is not None: self.prob_exponent = alpha
        if N is not None: self.N = N
        if W is not None: self.W = W
        
        # Create seed array
        seed_arr = np.arange(0, N_repeat * len(m_vals)).reshape(len(m_vals), N_repeat)
        group_name_arr = np.zeros((len(m_vals), N_repeat), dtype=object)
        print("Total number of iterations: ", len(m_vals) * N_repeat)
        # Loop over ds values, get the gname and then store the data for that gname
        for i, m in enumerate(m_vals):
            for j in range(N_repeat):
                self._set_seed(seed_arr[i, j])
                self.mutation_magnitude = m
                if linear_smin: self.salary_min = m / 10
                group_name_arr[i, j] = self._get_group_name()
                if store_data: self.store_data_in_group(print_info=False)
        
        # Reset values
        self.mutation_magnitude = current_m
        self._set_seed(seed=current_seed)
        self.prob_exponent = current_prob_exponent
        self.N = current_N
        self.W = current_W
        self.salary_min = current_smin
        
        if N_repeat == 1:
            return group_name_arr.flatten()
        
        return group_name_arr

    def generate_smin_arr_data(self, smin_vals, N_repeat, store_data=True, alpha=None, N=None, W=None):
        """Store data for different m values with N_repeat repeats or if store_data=False only return group names.

        Args:
            m_vals (_type_): _description_
            N_repeat (_type_): _description_

        Returns:
            np.ndarray: Group name arrays for ds values with repeats
        """
        # Get current values such that they can later be reset to these
        current_N = self.N
        current_W = self.W
        current_seed = self.seed
        current_prob_exponent = self.prob_exponent
        current_m = self.mutation_magnitude
        current_smin = self.salary_min
        
        # Set alpha, N and W if given, otherwise they are the default values
        if alpha is not None: self.prob_exponent = alpha
        if N is not None: self.N = N
        if W is not None: self.W = W
        
        # Create seed array
        seed_arr = np.arange(0, N_repeat * len(smin_vals)).reshape(len(smin_vals), N_repeat)
        group_name_arr = np.zeros((len(smin_vals), N_repeat), dtype=object)
        print("Total number of iterations: ", len(smin_vals) * N_repeat)
        # Loop over ds values, get the gname and then store the data for that gname
        for i, smin in enumerate(smin_vals):
            for j in range(N_repeat):
                self._set_seed(seed_arr[i, j])
                self.salary_min = smin
                group_name_arr[i, j] = self._get_group_name()
                if store_data: self.store_data_in_group(print_info=False)
        
        # Reset values
        self.mutation_magnitude = current_m
        self._set_seed(seed=current_seed)
        self.prob_exponent = current_prob_exponent
        self.N = current_N
        self.W = current_W
        self.salary_min = current_smin
        
        if N_repeat == 1:
            return group_name_arr.flatten()
        
        return group_name_arr


    def generate_ds_arr_data(self, ds_vals, N_repeat, store_data=True, alpha=None, N=None, W=None):
        """Store data for different ds values with N_repeat repeats or if store_data=False only return group names.

        Args:
            ds_vals (_type_): _description_
            N_repeat (_type_): _description_

        Returns:
            np.ndarray: Group name arrays for ds values with repeats
        """
        # Get current values such that they can later be reset to these
        current_N = self.N
        current_W = self.W
        current_seed = self.seed
        current_prob_exponent = self.prob_exponent
        current_smin = self.salary_min
        current_ds = self.ds
        
        # Set alpha, N and W if given, otherwise they are the default values
        if alpha is not None: self.prob_exponent = alpha
        if N is not None: self.N = N
        if W is not None: self.W = W
        
        # Create seed array
        seed_arr = np.arange(0, N_repeat * len(ds_vals)).reshape(len(ds_vals), N_repeat)
        group_name_arr = np.zeros((len(ds_vals), N_repeat), dtype=object)
        # Loop over ds values, get the gname and then store the data for that gname
        for i, ds in enumerate(ds_vals):
            for j in range(N_repeat):
                self._set_seed(seed_arr[i, j])
                self.ds = ds
                group_name_arr[i, j] = self._get_group_name()
                if store_data: self.store_data_in_group(print_info=False)
        
        # Reset values
        self.ds = current_ds
        self._set_seed(seed=current_seed)
        self.prob_exponent = current_prob_exponent
        self.N = current_N
        self.W = current_W
        
        return group_name_arr


    def generate_alpha_arr_data(self, alpha_vals, N_repeat, store_data=True, N=None, W=None):
        """Store data for different ds values with N_repeat repeats or if store_data=False only return group names.

        Args:
            alpha_vals (_type_): _description_
            N_repeat (_type_): _description_

        Returns:
            np.ndarray: Group name arrays for ds values with repeats
        """
        # Get current values such that they can later be reset to these
        current_N = self.N
        current_W = self.W
        current_seed = self.seed
        current_prob_exponent = self.prob_exponent
        
        # Set alpha, N and W if given, otherwise they are the default values
        if N is not None: self.N = N
        if W is not None: self.W = W
        
        # Create seed array
        seed_arr = np.arange(0, N_repeat * len(alpha_vals)).reshape(len(alpha_vals), N_repeat)
        group_name_arr = np.zeros((len(alpha_vals), N_repeat), dtype=object)
        # Loop over ds values, get the gname and then store the data for that gname
        for i, alpha in enumerate(alpha_vals):
            for j in range(N_repeat):
                self._set_seed(seed_arr[i, j])
                self.prob_exponent = alpha
                group_name_arr[i, j] = self._get_group_name()
                if store_data: self.store_data_in_group(print_info=False)
        
        # Reset values
        self._set_seed(seed=current_seed)
        self.prob_exponent = current_prob_exponent
        self.N = current_N
        self.W = current_W
        
        if N_repeat == 1:
            return group_name_arr.flatten()
        
        return group_name_arr
        
    

    def generate_ds_tensor_data(self, ds_vals, N_repeat, alpha_vals, N_vals, W_vals, store_data, time_steps=None) -> list:
        """Generate data for a tensor of ds values with repeats for each alpha and N, W values. Elementa in N and W are run pairwise together, e.g. N[0], W[0] are run together.

        Args:
            ds_vals (_type_): _description_
            N_repeat (_type_): _description_
            alpha_vals (_type_): _description_
            N_vals (_type_): _description_
            W_vals (_type_): _description_

        Returns:
            list: list of group name arrays
        """
        current_time_steps = self.time_steps
        # If a time value is given, add the skip values to it
        if time_steps is not None:
            self.time_steps = time_steps
        
        # Print the total number of iterations
        if store_data: print(f"Total number of iterations: {len(ds_vals) * len(alpha_vals) * len(N_vals)  * N_repeat}")
        # Create a list to store the group name arrays
        list_of_group_name_arr = []
        for alpha in alpha_vals:
            for N, W in zip(N_vals, W_vals):
                name_arr = self.generate_ds_arr_data(ds_vals, N_repeat, store_data=store_data, alpha=alpha, N=N, W=W) 
                list_of_group_name_arr.append(name_arr)
        
        # Reset time steps
        self.time_steps = current_time_steps
        
        return list_of_group_name_arr


    def generate_N_W_arr_data(self, N_values, ratio_vals, number_of_repeats, W_values=None, store_data=True, alpha=None, default_values={}):
        """Store data for three different scenarios: N variable and W constant; N constant and W variable;  N / W ratio.

        Args:
            variable_arr (_type_): The value of either N or W that is varied
            constant_value (_type_): The value of the constant variable
            number_of_repeats (_type_): Number of repeats for each variable value
            store_data (_type_): Whether to store the data or just return the group names
            alpha (_type_): The value of the alpha parameter. If None, use the default value
        Returns:
            np.ndarray: Group name arrays for ds values with repeats
        """
        # Get current values such that they can later be reset to these
        current_N = self.N
        current_W = self.W
        current_seed = self.seed
        current_prob_exponent = self.prob_exponent
        
        # Set alpha, N and W if given, otherwise they are the default values
        if alpha is not None: self.prob_exponent = alpha
        
        # If W_values is None, set it equal to N_values
        if W_values is None: W_values = N_values
        
        # If W_values is not a list or array, create a list with the same value for each N
        if not isinstance(W_values, (list, np.ndarray)):
            W_values = [W_values] * len(N_values)
        
        # Create seed array.
        # If there is no repeated measurements, the seed array is 1d, otherwise it is 2d
        no_repeated_measurements = number_of_repeats == 1
        most_values = N_values if len(N_values) > len(ratio_vals) else ratio_vals
        if no_repeated_measurements:
            seed_arr = np.arange(0, len(most_values))
        else:
            seed_arr = np.arange(0, number_of_repeats * len(most_values)).reshape(len(most_values), number_of_repeats)
        # Store gnames arrays
        N_gname_arr = np.zeros((len(N_values), number_of_repeats), dtype=object)
        ratio_gname_arr = np.zeros((len(ratio_vals), number_of_repeats), dtype=object)
        if store_data: print("Total number of iterations: ", (len(N_values) + len(ratio_vals)) * number_of_repeats)
        # Loop over ds values, get the gname and then store the data for that gname
        for i, N in enumerate(N_values):
            for j in range(number_of_repeats):
                if no_repeated_measurements: 
                    self._set_seed(seed_arr[i])
                else:
                    self._set_seed(seed_arr[i, j])
                self.N = N
                self.W = W_values[i]
                N_gname_arr[i, j] = self._get_group_name()
                if store_data: self.store_data_in_group(print_info=False)
        if store_data: print("Finished N variable, W constant")
        for i, ratio in enumerate(ratio_vals):
            for j in range(number_of_repeats):
                if no_repeated_measurements: 
                    self._set_seed(seed_arr[i])
                else:
                    self._set_seed(seed_arr[i, j])
                self.N = N_values[0]
                self.W = int(ratio * N_values[0])
                ratio_gname_arr[i, j] = self._get_group_name()
                if store_data: self.store_data_in_group(print_info=False)
        if store_data: print("Finished ratio")
        # Reset values
        self._set_seed(seed=current_seed)
        self.prob_exponent = current_prob_exponent
        self.N = current_N
        self.W = current_W
        
        if no_repeated_measurements:
            return N_gname_arr.flatten(), ratio_gname_arr.flatten()
        return N_gname_arr, ratio_gname_arr
    
    
    def generate_system_data(self, alpha_values, N_values, W_factor, store_data=True):
        """Store data for different alpha, N, W=W_factor * N values with N_repeat repeats or if store_data=False only return group names.

        Args:
            ds_vals (_type_): _description_
            N_repeat (_type_): _description_

        Returns:
            np.ndarray: Group name arrays for ds values with repeats
        """
        # Get current values such that they can later be reset to these
        current_N = self.N
        current_W = self.W
        current_seed = self.seed
        current_prob_exponent = self.prob_exponent
        
        # Create seed array
        seed_arr = np.arange(0, len(alpha_values))
        group_name_arr = np.zeros_like(seed_arr, dtype=object)
        for i, alpha, N in zip(np.arange(len(alpha_values)), alpha_values, N_values):
            self._set_seed(seed_arr[i])
            self.prob_exponent = alpha
            self.N = N
            self.W = int(W_factor * N)
            
            group_name_arr[i] = self._get_group_name()
            if store_data: self.store_data_in_group(print_info=False)
        
        # Reset values
        self._set_seed(seed=current_seed)
        self.prob_exponent = current_prob_exponent
        self.N = current_N
        self.W = current_W
        
        return group_name_arr
            
            
if __name__ == "__main__":
    number_of_companies = 10
    number_of_workers = 20
    salary_increase = 0.075
    interest_rate_free = 0.05
    mutation_magnitude = 0.01  # float, "spread" or "lastT"
    prob_exponent = 1
    update_methods = {"worker_update": "unlimited", 
                    "bankruptcy": "cannot_pay_salary",
                    "mutation": "constant"}
    time_steps = 10_000
    seed = 42
    run = RunWorkForce(number_of_companies, number_of_workers, salary_increase, interest_rate_free, mutation_magnitude, prob_exponent, update_methods, time_steps, seed)
    group_name = run.group_name

    run.store_data_in_group()
    print("Finished storing data")