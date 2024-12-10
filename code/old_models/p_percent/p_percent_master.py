import numpy as np
from tqdm import tqdm
from pathlib import Path
import h5py


class BankpPercent():
    def __init__(self, number_of_companies, money_to_production_efficiency, interest_rate_change_size, time_steps):
        self.N = number_of_companies
        self.alpha = money_to_production_efficiency
        self.rho = interest_rate_change_size
        self.time_steps = time_steps
        
        # Local paths for saving files
        file_path = Path(__file__)        
        self.dir_path = file_path.parent.parent.parent  # ./models/file
        self.dir_path_output = Path.joinpath(self.dir_path, "output")
        self.dir_path_image = Path.joinpath(self.dir_path, "images", "image_p_percent")
        self.group_name = f"Steps{self.time_steps}_N{self.N}_alpha{self.alpha}_rho{self.rho}"

        # Minimum values
        self.interest_rate_free_min = 1e-3        
        self.p_min = 1
        
        # Seed
        # np.random.seed(42)
        
    
    def _initialize_market_variables(self) -> None:    
        # Company variables
        self.p = self.p_min * np.ones(self.N, dtype=float)
        self.d = np.zeros(self.N, dtype=float)
        self.p_change = np.ones(self.N, dtype=float)

        # Bank variables
        self.interest_rate_free = 0.05
        self.interest_rate = self.interest_rate_free
        self.went_bankrupt_list = [1]
        

    def _initialize_hist_arrays(self) -> None:
        # Initialize hist arrays
        # Company
        self.p_hist = np.zeros((self.N, self.time_steps))
        self.d_hist = np.zeros((self.N, self.time_steps))
        self.average_p_change_hist = np.empty(self.time_steps)
        
        # Bank
        self.interest_rate_hist = np.zeros(self.time_steps, dtype=float)
        
        # Initial values
        self.p_hist[:, 0] = self.p
        self.d_hist[:, 0] = self.d
        self.average_p_change_hist[0] = np.mean(self.p_change)
        self.interest_rate_hist[0] = self.interest_rate


    def _transaction(self, buyer_idx, seller_idx) -> None:
        # Loan size using last transaction growth
        # loan_size = self.p[buyer_idx] * self.p_change[buyer_idx] / self.alpha

        # Loan size using mean
        loan_size = self.p[buyer_idx] * np.mean(self.p_change) / self.alpha
        
        # Clip loan size
        loan_size_clip = np.clip(loan_size, a_min=0, a_max=self.p[seller_idx])
        
        # Update values
        # Production and change in production
        p_old = self.p[buyer_idx]
        self.p[buyer_idx] += self.alpha * loan_size_clip
        self.p_change[buyer_idx] = (self.p[buyer_idx] - p_old) / p_old

        # Debt
        self.d[buyer_idx] += loan_size_clip
        self.d[seller_idx] -= loan_size_clip


    def _pay_interest(self) -> None:   
        positive_debt_idx = self.d > 0
        self.d[positive_debt_idx] *= 1 + self.interest_rate
    
    
    def _bankruptcy(self) -> None:
        # Find idx and number of bankrupt companies
        bankrupt_idx = np.where(self.p < self.interest_rate * self.d)[0]
        self.went_bankrupt = len(bankrupt_idx)
        self.went_bankrupt_list.append(self.went_bankrupt)
        
        # Start new companies in place of dead ones
        if self.went_bankrupt > 0:
            self.p[bankrupt_idx] = 1
            self.d[bankrupt_idx] = 0
    
            
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
        self.average_p_change_hist[time_step] = np.mean(self.p_change)

        # Bank variables
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
            self._adjust_interest_for_default()
            self._store_values_in_hist_arrays(time_step=i)
            
            
    def store_values(self, func_buyer_seller_idx) -> None:
        # Check if output directory exists
        self.dir_path_output.mkdir(parents=True, exist_ok=True)
        
        # File name and path
        file_name = "p_percent_simulation_data.h5"
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
        group.create_dataset("average_p_change", data=self.average_p_change_hist)
        # Bank variables
        group.create_dataset("interest_rate", data=self.interest_rate_hist)
        
        # Other
        group.create_dataset("went_bankrupt", data=self.went_bankrupt_list[1:])
        
        # Attributes
        group.attrs["alpha"] = self.alpha
        f.close()
        
        
# Define variables for other files to use
number_of_companies = 200
time_steps = 1000
money_to_production_efficiency = 1  # alpha
interest_rate_change_size = 0.02  # rho, percentage change in r

# Other files need some variables
bank_p_lim = BankpPercent(number_of_companies, money_to_production_efficiency, interest_rate_change_size, time_steps)

dir_path_output = bank_p_lim.dir_path_output
dir_path_image = bank_p_lim.dir_path_image
group_name = bank_p_lim.group_name

if __name__ == "__main__":
    print("You ran the wrong script :)")