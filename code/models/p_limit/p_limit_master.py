import numpy as np
from tqdm import tqdm
from pathlib import Path
import h5py
from typing import Callable


class BankpLim():
    def __init__(self, number_of_companies: int, money_to_production_efficiency: float,
                 interest_rate_change_size: float, time_steps: int) -> None:
        
        self.N = number_of_companies
        self.alpha = money_to_production_efficiency
        self.rho = interest_rate_change_size
        self.time_steps = time_steps
        
        # Local paths for saving files
        file_path = Path(__file__)
        self.dir_path = file_path.parent.parent.parent
        self.dir_path_output = Path.joinpath(self.dir_path, "output")
        self.dir_path_image = Path.joinpath(self.dir_path, "images", "image_p_limit")
        self.group_name = f"Steps{self.time_steps}_N{self.N}_alpha{self.alpha}_rho{self.rho}"
        
        # Minimum values
        self.beta_min = 1e-2
        self.interest_rate_min = 1e-3
        self.interest_rate_free_min = 1e-3
        
        # Seed
        np.random.seed(42)
        
    
    def _initialize_market_variables(self) -> None:
        # Company variables
        self.p = np.ones(self.N, dtype=float)
        self.d = np.zeros(self.N, dtype=float)
        
        # Bank variables
        self.d_bank = 0
        self.interest_rate_free = 0.05
        self.interest_rate = self.interest_rate_free
        self.went_bankrupt_list = [1]
        
        
    def _initialize_hist_arrays(self) -> None:
        # Initialize hist arrays
        # Company
        self.p_hist = np.zeros((self.N, self.time_steps))
        self.d_hist = np.zeros((self.N, self.time_steps))
        # Bank
        self.interest_rate_free_hist = np.zeros(self.time_steps, dtype=float)
        self.interest_rate_hist = np.zeros(self.time_steps, dtype=float)
        
        # Initial values
        self.p_hist[:, 0] = self.p
        self.d_hist[:, 0] = self.d
        
        # Bank
        self.interest_rate_free_hist[0] = self.interest_rate_free
        self.interest_rate_hist[0] = self.interest_rate
        
        self.average_income = 2.
        self.total_income = 0.
        
    
    def _transaction(self, buyer_idx: int, seller_idx: int) -> None:
        # Check if economy is healthy i.e. if money growth is larger than debt growth
        if self.alpha / 2 > self.interest_rate:
            # Find loan size and make sure it is larger than zero and smaller than seller's production
            loan_size = (self.average_income - self.p[buyer_idx]) / self.alpha
            loan_size_clipped = np.clip(a=loan_size, a_min=0., a_max=self.p[seller_idx])

            # Update values
            self.d[buyer_idx] += loan_size_clipped
            self.d[seller_idx] -= loan_size_clipped
            self.p[buyer_idx] += self.alpha * loan_size_clipped
            
            self.total_income += loan_size_clipped
    
    
    def _pay_interest(self) -> None:   
        positive_debt_idx = self.d > 0
        self.d[positive_debt_idx] *= 1 + self.interest_rate
    
    
    def _bankruptcy(self) -> None:
        # Find idx and number of bankrupt companies
        bankrupt_idx = np.where(self.p < self.d)[0]
        self.went_bankrupt = len(bankrupt_idx)
        self.went_bankrupt_list.append(self.went_bankrupt)
        
        # Start new companies in place of dead ones
        if self.went_bankrupt > 0:
            self.p[bankrupt_idx] = 1
            self.d[bankrupt_idx] = 0
    
    
    def _update_free_interest_rate(self) -> None:    
        # Compare average expenses and average income
        # Calculate total expenses
        total_expenses = self.interest_rate * np.sum(self.d[self.d > 0])
        average_expenses = total_expenses / self.N 
        self.average_income = self.total_income / self.N
        
        # Adjust free interest_rate
        negative_bias_correction = 1 / (1 - self.rho)
        if average_expenses > self.average_income:
            self.interest_rate_free *= (1 - self.rho)
        else:
            self.interest_rate_free *= (1 + negative_bias_correction * self.rho)
        
        self.interest_rate_free = np.maximum(self.interest_rate_free, self.interest_rate_free_min)

        # Reset total income for next iteration
        self.total_income = 0.
    
    
    def _probability_of_default(self, T=20):
        self.PD = np.mean(self.went_bankrupt_list[-T:]) / self.N
        
    
    def _adjust_interest_for_default(self) -> None:
        # Using the probability of default (synonymous with bankruptcy) to adjust the interest rate
        self._probability_of_default()
        self.interest_rate = (1 + self.interest_rate_free) / (1 - self.PD) - 1 
        self.interest_rate = np.maximum(self.interest_rate, self.interest_rate_min)
        
    
    def _store_values_in_hist_arrays(self, time_step: int) -> None:
        # Company variables
        self.p_hist[:, time_step] = self.p
        self.d_hist[:, time_step] = self.d
        # Bank variables
        self.interest_rate_free_hist[time_step] = self.interest_rate_free
        self.interest_rate_hist[time_step] = self.interest_rate
        
    
    def _simulation(self, func_buyer_seller_idx) -> None:
        # Initialize variables and hist arrays
        self._initialize_market_variables()
        self._initialize_hist_arrays()
        
        # Run simulation
        for i in tqdm(range(self.time_steps)):
            # Perform N transactions per time step
            for _ in range(self.N):
                # Select buyer and seller
                buyer_idx, seller_idx = func_buyer_seller_idx()
                self._transaction(buyer_idx, seller_idx)
                
            self._pay_interest()
            self._bankruptcy()
            self._update_free_interest_rate()
            self._adjust_interest_for_default()
            self._store_values_in_hist_arrays(time_step=i)
            
            
    def store_values(self, func_buyer_seller_idx) -> None:
        # Check if output directory exists
        self.dir_path_output.mkdir(parents=True, exist_ok=True)
        
        # File name and path
        file_name = "p_limit_simulation_data.h5"
        file_path = self.dir_path_output / file_name
        
        # If the exact filename already exists, open in write, otherwise in append mode
        f = h5py.File(file_path, "a")
        if self.group_name in list(f.keys()):
            f.close()
            f = h5py.File(file_path, "w")

        group = f.create_group(self.group_name)
        
        # Run simulation to get data
        self._simulation(func_buyer_seller_idx)
        
        # Store data in group
        # Company variables
        group.create_dataset("p", data=self.p_hist)
        group.create_dataset("d", data=self.d_hist)
        # Bank variables
        group.create_dataset("interest_rate_free", data=self.interest_rate_free_hist)
        group.create_dataset("interest_rate", data=self.interest_rate_hist)
        
        # Other
        group.create_dataset("went_bankrupt", data=self.went_bankrupt_list[1:])
        
        # Attributes
        group.attrs["alpha"] = self.alpha
        f.close()


# Define variables for other files to use
number_of_companies = 150
time_steps = 200
money_to_production_efficiency = 0.30  # alpha
interest_rate_change_size = 0.01  # rho, percentage change in r

# Other files need some variables
bank_p_lim = BankpLim(number_of_companies, money_to_production_efficiency, interest_rate_change_size, time_steps)

dir_path_output = bank_p_lim.dir_path_output
dir_path_image = bank_p_lim.dir_path_image
group_name = bank_p_lim.group_name

if __name__ == "__main__":
    print("You ran the wrong script :)")