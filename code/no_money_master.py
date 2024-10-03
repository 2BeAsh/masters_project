import numpy as np
from tqdm import tqdm
from pathlib import Path
import h5py
from typing import Callable
from itertools import product


class BankNoMoney():
    def __init__(self, number_of_companies: int, money_to_production_efficiency: float,
                 interest_rate_change_size: float, beta_mutation_size: float, beta_update_method: str,
                 interest_update_method: str, time_steps: int) -> None:
        
        # Check if given parameters are valid
        assert interest_update_method in ["bank_debt", "allow_negative", "random_walk", "loan_supply_demand"], "Invalid free interest update method"
        
        # Local paths for saving files
        file_path = Path(__file__)
        self.dir_path = file_path.parent
        self.dir_path_output = Path.joinpath(self.dir_path, "output")
        self.dir_path_image = Path.joinpath(self.dir_path, "image_no_money")

        # Create a list of parameter values to iterate over
        self.param_values = {
            'number_of_companies': number_of_companies if isinstance(number_of_companies, list) else [number_of_companies],
            'money_to_production_efficiency': money_to_production_efficiency if isinstance(money_to_production_efficiency, list) else [money_to_production_efficiency],
            'interest_rate_change_size': interest_rate_change_size if isinstance(interest_rate_change_size, list) else [interest_rate_change_size],
            'beta_mutation_size': beta_mutation_size if isinstance(beta_mutation_size, list) else [beta_mutation_size],
            'beta_update_method': beta_update_method if isinstance(beta_update_method, list) else [beta_update_method],
            'time_steps': time_steps if isinstance(time_steps, list) else [time_steps]
        }
        self.first_group_params = f"Steps{self.param_values['time_steps'][0]}_N{self.param_values['number_of_companies'][0]}_alpha{self.param_values['money_to_production_efficiency'][0]}_rho{self.param_values['interest_rate_change_size'][0]}_dbeta{self.param_values['beta_mutation_size'][0]}_betaUpdate{self.param_values['beta_update_method'][0]}"

        # Variables not meant for iterations
        self.interest_update_method = interest_update_method
        
        # Set seed
        np.random.seed(42)
        

    def _initialize_market_variables(self) -> None:
        # Company variables
        self.p = np.ones(self.N)
        self.d = np.zeros(self.N)
        self.beta = np.random.uniform(low=0, high=1, size=self.N)

        # Bank variables
        self.d_bank = 0
        self.interest_rate_free = 0.1
        self.interest_rate = self.interest_rate_free


    def _initialize_hist_arrays(self) -> None:
        # Company hist
        self.p_hist = np.empty((self.N, self.time_steps))
        self.d_hist = np.empty((self.N, self.time_steps))
        self.beta_hist = np.empty((self.N, self.time_steps))

        # Bank hist
        self.d_bank_hist = np.empty(self.time_steps)
        self.interest_rate_hist_free = np.empty(self.time_steps)
        self.interest_rate_hist = np.empty(self.time_steps)
        
        # Other
        self.supply_demand_list = []

        # Set initial values
        self.p_hist[:, 0] = self.p
        self.d_hist[:, 0] = self.d
        self.beta_hist[:, 0] = self.beta

        self.d_bank_hist[0] = self.d_bank
        self.interest_rate_hist_free[0] = self.interest_rate_free
        self.interest_rate_hist[0] = self.interest_rate


    def _transaction(self, buyer_idx: int, seller_idx: int) -> None:
        # Buyer takes loan
        buyer_loan_max = self.beta[buyer_idx] * self.p[buyer_idx] / self.interest_rate - self.d[buyer_idx]
        delta_debt = np.min((buyer_loan_max, self.p[seller_idx]))
        delta_debt = np.max((delta_debt, 0))  # Ensure positive

        # Update values
        self.d[buyer_idx] += delta_debt  # buyer takes loan
        # self.d_bank -= delta_debt  # bank gives loan and thus has a claim (negative loan) on the buyer that just took a loan  # OBS remove comment
        self.d[seller_idx] -= delta_debt  # seller gains money (negative loan)
        self.p[buyer_idx] += self.alpha * delta_debt  # buyer uses money spendt to raise production
        
        # Record the difference between the max buyer loan and p[seller_idx]
        self.supply_demand_list.append(buyer_loan_max - self.p[seller_idx])


    def _pay_interest(self) -> None:
        # Companies pays interest, bank receives interest
        self.d += self.d * self.interest_rate
        self.d_bank -= self.interest_rate * np.sum(self.d)


    def _mutate_beta(self, bankrupt_indices) -> None:
        # All methods add a mutation
        mutation = np.random.uniform(low=-self.beta_mutation_size, high=self.beta_mutation_size, size=self.went_bankrupt)

        if self.beta_update_method == "production":
            # Set all bankrupt companies' beta value to the beta value of the company with the highest production
            max_production_idx = np.argmax(self.p)
            self.beta[bankrupt_indices] = self.beta[max_production_idx] + mutation

        elif self.beta_update_method == "random":
            # Set all bankrupt companies' beta value to that of a random non-bankrupt company.
            # Get all non-bankrupt indices and corresponding beta values
            idx_not_bankrupt = np.where(~np.isin(np.arange(self.N), bankrupt_indices))[0]
            beta_not_bankrupt = self.beta[idx_not_bankrupt]
            # Each bankrupt company gets a random beta value from the non-bankrupt companies
            beta_not_bankrupt_randomly_sampled = np.random.choice(beta_not_bankrupt, size=self.went_bankrupt, replace=True)
            # Update bankrupt companies' beta values
            self.beta[bankrupt_indices] = beta_not_bankrupt_randomly_sampled + mutation


    def _bankruptcy(self) -> None:
        # Get bankrupt indices
        bankrupt_indices = np.where(self.p < self.d)[0]
        self.went_bankrupt = len(bankrupt_indices)  # Number of companies that went bankrupt
        # Only run if there are bankrupt companies
        if self.went_bankrupt > 0:
            # Set bankrupt companies' values to initial values
            self.p[bankrupt_indices] = 1
            self.d[bankrupt_indices] = 0
            # Find new beta values evolutionarily
            self._mutate_beta(bankrupt_indices)


    def _store_values_in_hist_arrays(self, time_step: int) -> None:
        # Company
        self.p_hist[:, time_step] = self.p
        self.d_hist[:, time_step] = self.d
        self.beta_hist[:, time_step] = self.beta

        # Bank
        self.d_bank_hist[time_step] = self.d_bank
        self.interest_rate_hist_free[time_step] = self.interest_rate_free
        self.interest_rate_hist[time_step] = self.interest_rate


    def _update_free_interest(self, time_step: int) -> None:
        delta_bank_debt = self.d_bank - self.d_bank_hist[time_step-1]
        delta_debt = self.d.sum() - self.d_hist[:, time_step-1].sum()  # Positive means more money to the bank, negative means bank loses money
        
        negative_bias_correction = 1 / (1 - self.rho)

        # One step model
        if self.interest_update_method == "allow_negative":
            rho = self.rho
            if self.one_step_model and self.d.sum() < 0:
                rho = - rho
            
            if delta_debt > 0:  # Bank gains money, decrease interest rate
                self.interest_rate_free -= rho
            else:  # Bank loses money, increase interest rate
                self.interest_rate_free += rho
        
        elif self.interest_update_method == "bank_debt":
            if delta_debt > 0:  # Bank gains money, decrease interest rate
                self.interest_rate_free *= (1 - self.rho)
            else:  # Bank loses money, increase interest rate
                self.interest_rate_free *= (1 + negative_bias_correction * self.rho)
                
        elif self.interest_update_method == "random_walk":
            q = np.random.uniform(low=0, high=1)
            if q < 0.5:
                self.interest_rate_free += self.rho
            else:
                self.interest_rate_free -= self.rho
                
        elif self.interest_update_method == "loan_supply_demand":
            # Supply and demand is determined by the number of people who could and could not take full loan
            # The median of the supply_demand_list is used to determine the direction of the interest rate change
            median_supply_demand = np.median(self.supply_demand_list)
            # If the median is positive, the buying power is too strong (i.e. the demand is too high), so the interest rate should increase and vice versa
            
            if median_supply_demand > 0:
                self.interest_rate_free *= (1 + negative_bias_correction * self.rho)
            else:
                self.interest_rate_free *= (1 - self.rho)
            # Reset the supply demand list
            self.supply_demand_list = []


    def _adjust_interest_for_default_probability(self) -> None:
        
        # Calculate probability of default
        self.PD = np.clip(a=self.went_bankrupt / self.N, a_min=0.01, a_max=0.99)  # Prevent division by zero in next step and that the interest rate becomes 0

        self.one_step_model = True
        if self.one_step_model:
            # One step interest formula adjustement
            self.interest_rate = (1 + self.interest_rate_free) / (1 - self.PD) - 1
        else:    
            # Adjust interest rate
            self.interest_rate = self.interest_rate_free * self.PD / ((1 - self.PD) * (1 + self.interest_rate_free))

        # self.interest_rate = np.max((self.interest_rate, 1e-2))


    def simulation(self, func_buyer_seller_idx) -> None:
        # Initialize
        self._initialize_market_variables()
        self._initialize_hist_arrays()

        # Run simulation
        for i in range(self.time_steps):
            for _ in range(self.N):
                buyer_idx, seller_idx = func_buyer_seller_idx()
                self._transaction(buyer_idx, seller_idx)

            # Pay interest
            self._pay_interest()
            # Bankruptcy and beta mutation
            self._bankruptcy()
            # Bank updates interest and takes probability of default into account
            self._update_free_interest(time_step=i)
            self._adjust_interest_for_default_probability()
            # Store values
            self._store_values_in_hist_arrays(time_step=i)


    def store_values(self, func_buyer_seller_idx: Callable) -> None:
        # Ensure the output directory exists
        self.dir_path_output.mkdir(parents=True, exist_ok=True)

        # Define the file name based on the parameters
        file_name = "no_money_data.h5"
        file_path = self.dir_path_output / file_name

        # Open the file in append mode
        with h5py.File(file_path, 'w') as f:
            # Create a list of all parameter combinations
            param_combinations = list(zip(
            self.param_values['number_of_companies'],
            self.param_values['money_to_production_efficiency'],
            self.param_values['interest_rate_change_size'],
            self.param_values['beta_mutation_size'],
            self.param_values['beta_update_method'],
            self.param_values['time_steps']
            ))

            # Iterate over all combinations of parameter values with tqdm
            for N, alpha, rho, beta_mutation_size, beta_update_method, time_steps in tqdm(param_combinations, desc="Parameter combinations"):
                # Update instance variables
                self.N = N
                self.alpha = alpha
                self.rho = rho
                self.beta_mutation_size = beta_mutation_size
                self.beta_update_method = beta_update_method
                self.time_steps = time_steps

                # Create a group for the current parameter combination
                group_name = f"Steps{time_steps}_N{N}_alpha{alpha}_rho{rho}_dbeta{beta_mutation_size}_betaUpdate{beta_update_method}"

                if group_name in f:
                    print("This dataset already exists")
                    print("Dataset: ", group_name)
                    return

                group = f.create_group(group_name)

                # Run the simulation to generate data hist arrays
                self.simulation(func_buyer_seller_idx)

                # Store the hist arrays in the group
                # Company
                group.create_dataset('p_hist', data=self.p_hist)
                group.create_dataset('d_hist', data=self.d_hist)
                group.create_dataset('beta_hist', data=self.beta_hist)
                # Bank
                group.create_dataset('d_bank_hist', data=self.d_bank_hist)
                group.create_dataset('interest_rate_hist_free', data=self.interest_rate_hist_free)
                group.create_dataset('interest_rate_hist', data=self.interest_rate_hist)
                # Save current variables as attributes
                group.attrs['N'] = N
                group.attrs['alpha'] = alpha
                group.attrs['rho'] = rho
                group.attrs['beta_mutation_size'] = beta_mutation_size
                group.attrs['beta_update_method'] = beta_update_method
                group.attrs['time_steps'] = time_steps
                # Print progress
                print(f"Stored values for {group_name}")

        print("Finished storing values")

        self.latest_run_group = group_name


# Define variables for other files to use
number_of_companies = 100
time_steps = 100
money_to_production_efficiency = 0.05  # Alpha
interest_rate_change_size = 0.01
beta_mutation_size = 0.1
beta_update_method = "production"  # "production" or "random"
interest_update_method = "random_walk"  # "allow_negative", "bank_debt", "random_walk", "loan_supply_demand"

# Other files need some variables
bank_no_money = BankNoMoney(number_of_companies, money_to_production_efficiency, interest_rate_change_size,
            beta_mutation_size, beta_update_method, interest_update_method, time_steps)

first_group_params = bank_no_money.first_group_params
dir_path_output = bank_no_money.dir_path_output
dir_path_image = bank_no_money.dir_path_image

if __name__ == "__main__":
    print("You ran the wrong script :\)")