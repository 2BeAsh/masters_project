import numpy as np
from tqdm import tqdm
from pathlib import Path
import h5py


class DebtDeflation():
    def __init__(self, N, time_steps, alpha, seed=None):
        self.N = N
        self.time_steps = time_steps
        self.alpha = alpha
        self.seed = seed
        
        # Local paths for saving files
        file_path = Path(__file__)
        self.dir_path = file_path.parent.parent.parent
        self.dir_path_output = Path.joinpath(self.dir_path, "output")
        self.dir_path_image = Path.joinpath(self.dir_path, "images", "OG_p_d_m")
        self.dir_path_image.mkdir(parents=True, exist_ok=True)
        self.group_name = f"Steps{self.time_steps}_N{self.N}_alpha{self.alpha}_seed{self.seed}"
        
        # Seed
        np.random.seed(self.seed)  
    
    
    def _initialize_market_variables(self):
        # System variables
        self.beta_min = 1e-3
        self.interest_rate_free = 0.05
        self.interest_rate = 0.05
        
        # Company values
        self.m = np.ones(self.N, dtype=float)
        self.d = np.zeros(self.N, dtype=float)
        self.p = np.ones(self.N, dtype=float)
        self.beta = np.random.uniform(self.beta_min, 1, self.N)
        
        # Initial values
        self.PD = 0.
        self.went_bankrupt = 0
        
        
    def _initialize_hist_arrays(self):
        # Company
        self.m_hist = np.zeros((self.N, self.time_steps), dtype=float)
        self.d_hist = np.zeros((self.N, self.time_steps), dtype=float)
        self.p_hist = np.zeros((self.N, self.time_steps), dtype=float)
        self.beta_hist = np.zeros((self.N, self.time_steps), dtype=float)
        
        # System
        self.interest_rate_hist = np.zeros(self.time_steps, dtype=float)
        self.interest_rate_free_hist = np.zeros(self.time_steps, dtype=float)
        self.went_bankrupt_hist = np.zeros(self.time_steps, dtype=int)
        
        # Initial values
        self.m_hist[:, 0] = self.m * 1
        self.d_hist[:, 0] = self.d * 1
        self.p_hist[:, 0] = self.p * 1
        self.beta_hist[:, 0] = self.beta * 1
        
        self.interest_rate_hist[0] = self.interest_rate * 1
        self.interest_rate_free_hist[0] = self.interest_rate_free * 1
        self.went_bankrupt_hist[0] = 1


    def _transaction(self):
        for _ in range(self.N):
            
            # Get indices and take loan if needed
            buyer_idx, seller_idx = self._buyer_seller_idx()
            self._take_loan(buyer_idx, seller_idx)
            
            # Update values
            B_min = 0
            B_max = self.p[seller_idx]
            B = np.clip(self.m[buyer_idx], B_min, B_max)
            self.m[buyer_idx] -= B
            self.m[seller_idx] += B
            self.p[buyer_idx] += self.alpha * B
            
    
    
    def _buyer_seller_idx(self):
        # Draw random buyer, then draw seller from the remaining companies
        buyer_idx = np.random.randint(0, self.N)
        available_seller_idx = np.arange(self.N)[np.arange(self.N)!=buyer_idx]
        seller_idx = np.random.choice(available_seller_idx)
        return buyer_idx, seller_idx
                

    def _take_loan(self, buyer_idx, seller_idx):
        if self.m[buyer_idx] < self.p[seller_idx]:
            # Find loan size
            loan_min = 0
            loan_max = self.p[seller_idx] - self.m[buyer_idx]
            loan_x = (self.beta[buyer_idx] * self.p[buyer_idx] - self.interest_rate * self.d[seller_idx]) / (self.interest_rate - self.alpha * self.beta[buyer_idx])
            loan_size = np.clip(loan_x, loan_min, loan_max)
            
            # Update values
            self.m[buyer_idx] += loan_size
            self.d[buyer_idx] += loan_size  


    def _pay_interest(self):
        self.d += self.d * self.interest_rate


    def _negative_money_to_loan(self):
        """If any company has entered negative money, take a loan to correct it.
        """
        # Find companies with negative money and take debt to get the money to a small positive value to make log-plotting easier
        idx_negative_money = np.where(self.m <= 0)
        debt = np.abs(self.m[idx_negative_money]) + 0.01  
        # Update values
        self.m[idx_negative_money] += debt
        self.d[idx_negative_money] += debt


    def _bankruptcy(self):
        # Find companies with p < rd, which is the bankruptcy condition.
        # Set their values to the initial values
        bankrupt_idx = np.where(self.p < self.d * self.interest_rate)[0]
        self.went_bankrupt = bankrupt_idx.size
        
        if self.went_bankrupt > 0:
            self.p[bankrupt_idx] = 1
            self.m[bankrupt_idx] = 1
            self.d[bankrupt_idx] = 0
            
            # Find beta using evolutionary method
            self._mutate_beta(bankrupt_idx)
            
    
    def _mutate_beta(self, bankrupt_idx):
        """Draw beta values randomly from the non-bankrupt companies.
        Do this by drawing the indices of the non-bankrupt companies and get beta values from that
        New beta values is then these chosen beta values plus/minus a small change"""
        surviving_idx = np.arange(self.N)[~np.isin(np.arange(self.N), bankrupt_idx)]
        random_idx_from_survivors = np.random.choice(surviving_idx, size=bankrupt_idx.size, replace=True)
        beta_from_survivors = self.beta[random_idx_from_survivors]
        
        # Mutate beta and ensure does not go below minimum value
        self.beta[bankrupt_idx] = beta_from_survivors + np.random.uniform(-0.1, 0.1, size=bankrupt_idx.size)
        self.beta[bankrupt_idx] = np.maximum(self.beta[bankrupt_idx], self.beta_min)
        
    
    def _change_free_interest_rate(self, time_step): 
        # Calculate economy health. If doing good, raise interest rate, if doing bad, lower interest rate.
        # Economy health is change in money minus change in debt
        change_in_money = self.m - self.m_hist[:, time_step-1]
        change_in_debt = self.d - self.d_hist[:, time_step-1]
        # change_in_economy_health = np.sum(change_in_money - change_in_debt)
        change_in_economy_health = np.sum(change_in_debt)
        
        interest_change_size = 0.025
        positive_correction_factor = 1 / (1 - interest_change_size)
        if change_in_economy_health >= 0:
            self.interest_rate_free *= (1 + interest_change_size * positive_correction_factor)
        else:
            self.interest_rate_free *= (1 - interest_change_size)

        self.interest_rate_free = np.maximum(self.interest_rate_free, 1e-4)

    
    def _adjust_interest_for_default(self):
        PD = self.went_bankrupt / self.N  # Probability of default/bankruptcy
        PD = np.minimum(PD, 0.999)  # Avoid division by zero
        self.interest_rate = (1 + self.interest_rate_free) / (1 - PD) - 1  
        self.interest_rate = np.max((self.interest_rate, 1e-4))
    
    
    def _store_values_in_hist_arrays(self, time_step):
        self.m_hist[:, time_step] = self.m * 1
        self.d_hist[:, time_step] = self.d * 1
        self.p_hist[:, time_step] = self.p * 1
        self.beta_hist[:, time_step] = self.beta * 1
        self.went_bankrupt_hist[time_step] = self.went_bankrupt * 1
        self.interest_rate_hist[time_step] = self.interest_rate * 1
        self.interest_rate_free_hist[time_step] = self.interest_rate_free * 1
        
        
    def _simulation(self):
        self._initialize_market_variables()
        self._initialize_hist_arrays()
        
        for i in tqdm(range(1, self.time_steps)):
            self._transaction()
            self._pay_interest()
            self._negative_money_to_loan()
            self._bankruptcy()
            self._change_free_interest_rate(time_step=i)
            self._adjust_interest_for_default()
            self._store_values_in_hist_arrays(time_step=i)
            
            
    def store_data(self):
        # Get data
        self._simulation()
        # Check if output directiory excists
        self.dir_path_output.mkdir(parents=True, exist_ok=True)
        # File name and path
        file_name = "DebtDeflation.h5"
        file_path = self.dir_path_output / file_name
        
        with h5py.File(Path.joinpath(file_path), "a") as f:
            # If the group already exists, remove it
            if self.group_name in f:
                del f[self.group_name]
            
            group = f.create_group(self.group_name)
            # Dataset
            group.create_dataset("m", data=self.m_hist)
            group.create_dataset("d", data=self.d_hist)
            group.create_dataset("p", data=self.p_hist)
            group.create_dataset("beta", data=self.beta_hist)
            group.create_dataset("went_bankrupt", data=self.went_bankrupt_hist)
            group.create_dataset("interest_rate", data=self.interest_rate_hist)
            group.create_dataset("interest_rate_free", data=self.interest_rate_free_hist)
            # Attributes
            group.attrs["alpha"] = self.alpha
            group.attrs["interest_rate_free"] = self.interest_rate_free
            group.attrs["seed"] = self.seed


# Define variables
N = 200
time_steps = 5000
alpha = 0.075
interest_rate_free = 0.05
seed = 42

debtdeflation = DebtDeflation(N, time_steps, alpha, seed)

dir_path_output = debtdeflation.dir_path_output    
dir_path_image = debtdeflation.dir_path_image    
group_name = debtdeflation.group_name

# Store values
if __name__ == "__main__":
    debtdeflation.store_data()
    print("Stored values")