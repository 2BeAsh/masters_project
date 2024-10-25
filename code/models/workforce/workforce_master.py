import numpy as np
from tqdm import tqdm
from pathlib import Path
import h5py


class Workforce():
    def __init__(self, number_of_companies, number_of_workers, interest_rate_change_size, salary_factor, time_steps):
        self.N = number_of_companies
        self.W = number_of_workers
        self.rho = interest_rate_change_size
        self.sigma = salary_factor
        self.time_steps = time_steps
        
        # Local paths for saving files
        file_path = Path(__file__)
        self.dir_path = file_path.parent.parent.parent
        self.dir_path_output = Path.joinpath(self.dir_path, "output")
        self.dir_path_image = Path.joinpath(self.dir_path, "images", "image_workforce")
        self.dir_path_image.mkdir(parents=True, exist_ok=True)
        self.group_name = f"Steps{self.time_steps}_N{self.N}_W{self.W}_rho{self.rho}"
        
        # Seed
        # np.random.seed(42)   


    def _initialize_market_variables(self) -> None:    
        # Company variables
        self.p = np.ones(self.N, dtype=float)
        self.d = np.zeros(self.N, dtype=float)
        
        # Worker variables
        self.unemployed = self.W - self.N  # Every company starts with one employee
        
        # Bank variables
        self.interest_rate_free = 0.05
        self.interest_rate = self.interest_rate_free
        self.went_bankrupt_list = [1]


    def _initialize_hist_arrays(self) -> None:
        # Initialize hist arrays
        # Company
        self.p_hist = np.zeros((self.N, self.time_steps))
        self.d_hist = np.zeros((self.N, self.time_steps))
        
        self.interest_rate_hist = np.zeros(self.time_steps, dtype=float)
        
        # Initial values
        self.p_hist[:, 0] = self.p
        self.interest_rate_hist[0] = self.interest_rate


    def _hire_and_fire(self, time_step) -> None:
        # Repeat the hire/fire process N times to allow each company on average to hire/fire once
        for _ in range(self.N):
            # Find random company
            idx = np.random.randint(0, self.N)
            # Check if it made profit last time step
            change_in_debt = self.d_hist[idx, time_step - 1] - self.d_hist[idx, time_step - 2] 
            # Hire or fire
            if change_in_debt < 0 and self.unemployed > 0:  # Reduced debt, hire
                self.p[idx] += 1
                self.unemployed -= 1
            elif change_in_debt > 0 and self.p[idx] > -1:  # Increased debt, fire. 
                self.p[idx] -= 1
                self.unemployed += 1


    def _sell(self):
        for _ in range(self.N):
            # Choose random company
            idx = np.random.randint(0, self.N)
            # Gain money from selling to workforce
            self.employed = self.W - self.unemployed
            self.d[idx] -= self.p[idx] * self.employed / self.W


    def _pay_interest(self) -> None:   
        positive_debt_idx = self.d > 0
        self.d[positive_debt_idx] *= 1 + self.interest_rate
    
    
    def _pay_salaries(self):
        self.d += self.sigma * self.p * self.employed / self.W
    
    
    def _bankruptcy(self) -> None:
        # Bankrupt companies satisfy that p <= 0 and d > 0
        all_who_went_bankrupt_idx = np.logical_and(self.p <= 0, self.d > 0)
        self.went_bankrupt = len(all_who_went_bankrupt_idx)
        self.went_bankrupt_list.append(self.went_bankrupt)
        
        # Start new companies in place of dead ones, but only if there are workers available
        for i in range(self.went_bankrupt):  # Going through them from low to high brings inbalance
            bankrupt_i = all_who_went_bankrupt_idx[i]
            if self.unemployed > 0:
                self.p[bankrupt_i] = 1
                self.d[bankrupt_i] = 0
                self.unemployed -= 1
   
   
    def _probability_of_default(self, T=20):
        self.PD = np.mean(self.went_bankrupt_list[-T:]) / self.N
        
    
    def _adjust_interest_for_default(self) -> None:
        # Using the probability of default (synonymous with bankruptcy) to adjust the interest rate
        self._probability_of_default()
        self.interest_rate = (1 + self.interest_rate_free) / (1 - self.PD) - 1 
        
    
    def _store_values_in_hist_arrays(self, time_step: int) -> None:
        # Company variables
        self.p_hist[:, time_step] = self.p
        self.d_hist[:, time_step] = self.d

        # Bank variables
        self.interest_rate_hist[time_step] = self.interest_rate
        
    def _simulation(self):
        # Initialize variables and hist arrays
        self._initialize_market_variables()
        self._initialize_hist_arrays()
        
        # Run simulation
        for i in tqdm(range(self.time_steps)):
            self._hire_and_fire(time_step=i)
            self._sell()
            self._pay_interest()
            self._pay_salaries()
            self._bankruptcy()
            self._adjust_interest_for_default()
            self._store_values_in_hist_arrays(time_step=i)                


    def store_values(self) -> None:
        # Check if output directory exists
        self.dir_path_output.mkdir(parents=True, exist_ok=True)
        
        # File name and path
        file_name = "workforce_simulation_data.h5"
        file_path = self.dir_path_output / file_name
        
        # If the exact filename already exists, open in write, otherwise in append mode
        f = h5py.File(file_path, "a")
        if self.group_name in list(f.keys()):
            f.close()
            f = h5py.File(file_path, "w")

        group = f.create_group(self.group_name)
        
        # Run simulation to get data
        self._simulation()
        
        # Store data in group
        # Company variables
        group.create_dataset("p", data=self.p_hist)
        group.create_dataset("d", data=self.d_hist)
        # Bank variables
        group.create_dataset("interest_rate", data=self.interest_rate_hist)
        
        # Other
        group.create_dataset("went_bankrupt", data=self.went_bankrupt_list[1:])
        
        # Attributes
        group.attrs["W"] = self.W
        group.attrs["sigma"] = self.sigma
        f.close()

            
# Define variables for other files to use
number_of_companies = 100
number_of_workers = 1000
time_steps = 1000
interest_rate_change_size = 0.02  # rho, percentage change in r
salary_factor = 0.5

# Other files need some variables
workforce = Workforce(number_of_companies, number_of_workers, interest_rate_change_size, salary_factor, time_steps)

dir_path_output = workforce.dir_path_output
dir_path_image = workforce.dir_path_image
group_name = workforce.group_name

if __name__ == "__main__":
    print("You ran the wrong script :)")