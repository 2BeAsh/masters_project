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


    def store_data_in_group(self):
        # Run simulation to get data
        self.group_name = self._get_group_name()
        print(f"Storing data in {self.group_name}")
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