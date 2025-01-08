import numpy as np
from pathlib import Path
import h5py
from define_methods import MethodsWorkForce


class RunWorkForce(MethodsWorkForce):
    def __init__(self, number_of_companies, number_of_workers, salary_increase, interest_rate_free, mutation_magnitude, prob_exponent, update_methods, time_steps, seed):
        # Inherit from MethodsWorkForce which inherits from WorkForce
        self.worker_update_method = update_methods["worker_update"]
        self.prob_exponent = prob_exponent
        if self.worker_update_method == "unlimited":
            number_of_workers = number_of_companies
        self.rf_name = interest_rate_free
        
        super().__init__(number_of_companies, number_of_workers, salary_increase, interest_rate_free, mutation_magnitude, prob_exponent, update_methods, time_steps, seed)
        # Local paths for saving files
        file_path = Path(__file__)
        self.dir_path = file_path.parent.parent.parent
        self.dir_path_output = Path.joinpath(self.dir_path, "output")
        self.dir_path_image = Path.joinpath(self.dir_path, "images", "common")
        self.dir_path_image.mkdir(parents=True, exist_ok=True)
        # File name and path
        file_name = "common.h5"
        self.file_path = self.dir_path_output / file_name


    def store_data_in_group(self):
        # Run simulation to get data
        self.group_name = self._get_group_name()
        print(f"Storing data in {self.group_name}")
        self._simulation()
                
        with h5py.File(self.file_path, "a") as file:
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
            
            
    def repeated_m_runs(self, N_repeat, m_values):
        self.group_name = self._get_group_name()
        print(f"Storing repeated m runs in {self.group_name}")
        mean_salary_arr = np.zeros((len(m_values), N_repeat, self.time_steps))
        total_iterations = len(m_values) * N_repeat
        for i, m in enumerate(m_values):
            for j in range(N_repeat):
                print(f"iteration: {i*N_repeat + j+1}/{total_iterations}")
                self.mutation_magnitude = m
                self._simulation()
                mean_salary = np.mean(self.s_hist, axis=0)  # Over companies
                mean_salary_arr[i, j, :] = mean_salary
                
        # Save data            
        with h5py.File(self.file_path, "a") as file:
            try:
                data_group = file[self.group_name]
            except KeyError:
                data_group = file.create_group(self.group_name)
            if "repeated_m_runs" in data_group:
                del data_group["repeated_m_runs"]
            data_group.create_dataset("repeated_m_runs", data=mean_salary_arr)
            data_group.attrs["m_repeated"] = m_values
        

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
file_path = run.file_path
group_name = run.group_name
dir_path_image = run.dir_path_image

if __name__ == "__main__":
    run.store_data_in_group()
    print("Finished storing data")