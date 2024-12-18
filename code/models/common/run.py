from pathlib import Path
import h5py
from define_methods import MethodsWorkForce


class RunWorkForce(MethodsWorkForce):
    def __init__(self, number_of_companies, number_of_workers, salary_increase, interest_rate_free, mutation_magnitude, worker_update_method, time_steps, seed):
        # Inherit from MethodsWorkForce which inherits from WorkForce
        self.worker_update_method = worker_update_method
        if worker_update_method == "unlimited":
            number_of_workers = number_of_companies
        self.rf_name = interest_rate_free
        super().__init__(number_of_companies, number_of_workers, salary_increase, interest_rate_free, mutation_magnitude, worker_update_method, time_steps, seed)
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
        print(f"Storing data in {self.group_name}")
        self._simulation()
                
        with h5py.File(self.file_path, "a") as f:
            # If the group already exists, delete it
            if self.group_name in f:
                del f[self.group_name]
                
            # Create group
            group = f.create_group(self.group_name)
            # Store data in group
            # Company
            group.create_dataset("w", data=self.w_hist)
            group.create_dataset("d", data=self.d_hist)
            group.create_dataset("s", data=self.s_hist)
            # System
            group.create_dataset("r", data=self.r_hist)
            group.create_dataset("went_bankrupt", data=self.went_bankrupt_hist)
            group.create_dataset("mu", data=self.mu_hist)
            group.create_dataset("mutations", data=self.mutations_hist)
            # Attributes
            group.attrs["N"] = self.N
            group.attrs["W"] = self.W
            group.attrs["ds"] = self.ds
            group.attrs["rf"] = self.rf
            group.attrs["m"] = self.mutation_magnitude
            

number_of_companies = 10
number_of_workers = 20
salary_increase = 0.075
interest_rate_free = 0.05
mutation_magnitude = 0.01  # float, "spread" or "lastT"
worker_update_method = "limited"  # "limited" or "unlimited"
time_steps = 10_000
seed = 42
run = RunWorkForce(number_of_companies, number_of_workers, salary_increase, interest_rate_free, mutation_magnitude, worker_update_method, time_steps, seed)
file_path = run.file_path
group_name = run.group_name
dir_path_image = run.dir_path_image

if __name__ == "__main__":
    run.store_data_in_group()
    print("Finished storing data")