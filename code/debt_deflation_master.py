import numpy as np
from tqdm import tqdm
from pathlib import Path


class DebtDeflation():
    def __init__(self, number_of_companies: int, money_to_production_efficiency: float, real_interest_rate: float, buy_fraction: float, equilibrium_distance_fraction: float, include_debt: bool, time_steps: int):
        """Initializer

        Args:
            number_of_companies (int): _description_
            money_to_production_efficiency (float): _description_
            loan_probability (float): _description_
            interest_rate (float): _description_
            buy_fraction (float): _description_
            equilibrium_distance_fraction (float): _description_
            time_steps (int): _description_
        """
        self.N = number_of_companies
        self.alpha = money_to_production_efficiency  # Money to production efficiency
        self.real_interest_rate = real_interest_rate  # Interest rate
        self.buy_fraction = buy_fraction  # When doing a transaction, how large a fraction of min(seller production, buyer money) is used.
        self.epsilon = equilibrium_distance_fraction  # In inflation updates, the fraction the system goes toward equilibrlium.
        self.include_debt = include_debt
        self.time_steps = time_steps  # Steps taken in system evolution.
        
        # Local paths for saving files.
        file_path = Path(__file__)
        self.dir_path = file_path.parent
        self.dir_path_output = Path.joinpath(self.dir_path, "output")
        self.dir_path_image = Path.joinpath(self.dir_path, "image")
        self.file_parameter_addon_base = f"Steps{self.time_steps}_Companies{self.N}_Interest{self.real_interest_rate}_Efficiency{self.alpha}_EquilibriumStep{self.epsilon}"
        
        # Initial parameter values
        self.inflation_rate = 0  # Initially P = M, meaning no inflation
        self.r = self.real_interest_rate
        

    def _initial_market(self) -> None:
        """Initialize market.
        Production = 1, debt = 0, money = 1
        """
        # Companies
        self.production = np.ones(self.N)
        self.debt = np.zeros(self.N)
        self.money = np.ones(self.N)
    
    
    def _simple_loan(self, buyer_idx, seller_idx) -> None:
        # Calculate loan size
        loan_size = self.production[seller_idx] - self.money[buyer_idx]
        # Update values
        self.money[buyer_idx] += loan_size
        self.debt[buyer_idx] += loan_size


    def _smart_loan(self, buyer_idx, seller_idx) -> None:
        # Calculate loan size
        production_money_difference = self.production[seller_idx] - self.money[buyer_idx]
        production_debt_difference = np.max((self.production[buyer_idx] - self.r * self.debt[buyer_idx], 0))  # Production is ideal money gained, r * d is money lost. Cannot be negative, so max( ... , 0)
        # loan_size = np.min([production_money_difference, production_debt_difference])  # OBS Might uncomment
        loan_size = production_debt_difference
        # Update values
        self.money[buyer_idx] += loan_size
        self.debt[buyer_idx] += loan_size
        
        
    def _transaction(self, buyer_idx, seller_idx) -> None:
        """First check if the buyer needs to take a loan to match the sellers production, then make transaction and update values accordingly.

        Args:
            buyer_idx (int): Index of the buying company
            seller_idx (int): Index of the selling company
        """
        # If the buyer's money is less than the seller's production, 
        # the buyer takes a loan to try and match the production. 
        # The buyer cannot take a loan larger than its company size
        if self.money[buyer_idx] < self.production[seller_idx] and self.include_debt:
            self._smart_loan(buyer_idx, seller_idx)
            # self._simple_loan(buyer_idx, seller_idx)
            
        amount_bought = self.buy_fraction * np.min([self.production[seller_idx], self.money[buyer_idx]])
        amount_bought = np.max((amount_bought, 0))  # Money can go negative if the buyer cannot take a large enough loan to get to positive money
        
        # Update values
        self.money[seller_idx] += amount_bought
        self.money[buyer_idx] -= amount_bought
        self.production[buyer_idx] += self.alpha * amount_bought
    
    
    def _repay_debt(self) -> None:
        """Use money to pay off debt on the loans"""
        # Find the companies with positive money, and their money and debt, as negative money cannot pay off debt
        positive_idx = np.logical_and(self.money>0, self.debt>0)
        money_positive = self.money[positive_idx]
        debt_positive = self.debt[positive_idx]
                
        # Update values
        self.debt[positive_idx] = np.maximum(debt_positive - money_positive, 0)
        self.money[positive_idx] = np.maximum(money_positive - debt_positive, 0)

    
    def _pay_interest(self) -> None:
        """Companies pay interest on their debt. Update money accordingly.
        """
        self.money -= self.r * self.debt
    
    
    def _bankruptcy_check(self):
        """Bankrupt if p + m < d. If goes bankrupt, start a new company in its place with initial values.
        Intuition is that when m=0, you must have high enough production to pay off the debt in one transaction i.e. p > d"""
        # Find all companies who have gone bankrupt
        bankrupt_idx = np.where(self.production + self.money < self.debt * self.r)
        
        # Set their values to the initial values
        self.money[bankrupt_idx] = 1.
        self.debt[bankrupt_idx] = 0.
        self.production[bankrupt_idx] = 1.


    def _inflation(self):
        """Update the money according to the inflation rule:
        mi <- mi + epsilon * (P / M - 1) * mi

        Args:
            epsilon (float): How much of the distance to the equilibrium the step covers.
        """
        # Find inflation rate
        supply = self.production.sum()
        demand = self.money.sum()
        self.inflation_rate = self.epsilon * (supply / demand - 1)
        # Perform the update on money
        self.money += self.inflation_rate * self.money


    def _nominal_interest_rate(self) -> None:
        """Update the nominal interest rate r = gamma + pi + gamma * pi,
        where gamma is the real interest rate and pi is the inflation rate.
        """
        self.r = self.real_interest_rate + self.inflation_rate + self.real_interest_rate * self.inflation_rate    


    def _data_to_file(self) -> None:
        # Collect data to store in one array
        all_data = np.empty((self.N, self.time_steps, 3))
        all_data[:, :, 0] = self.production_hist
        all_data[:, :, 1] = self.debt_hist
        all_data[:, :, 2] = self.money_hist

        filename = Path.joinpath(self.dir_path_output, self.file_parameter_addon + ".npy")
        np.save(filename, arr=all_data)
        
        # Save inflation rate to another file
        filename_inflation = Path.joinpath(self.dir_path_output, "inflation_" + self.file_parameter_addon + ".npy")
        np.save(filename_inflation, arr=self.inflation_rate_hist)
    
    
    def simulation(self, func_buyer_seller_idx) -> None:
        """Run the simulation and store results in a file.

        Args:
            func_buyer_seller_idx (function): Choice of buyer and seller idx e.g. well mixed or 1d neighbours
        """
        # Initialize market
        self._initial_market()
        # History and its first value
        self.production_hist = np.zeros((self.N, self.time_steps))
        self.debt_hist = np.zeros_like(self.production_hist)
        self.money_hist = np.zeros_like(self.production_hist)
        self.inflation_rate_hist = np.zeros(self.time_steps)
        self.production_hist[:, 0] = self.production
        self.debt_hist[:, 0] = self.debt
        self.money_hist[:, 0] = self.money
        self.inflation_rate_hist[0] = self.inflation_rate
        
        # Time evolution
        for i in tqdm(range(1, self.time_steps)):
            # Make N transactions
            for _ in range(self.N):
                # Pick buyer and seller
                buyer_idx, seller_idx = func_buyer_seller_idx()
                self._transaction(buyer_idx, seller_idx)
            # Pay rent and check for bankruptcy
            if self.include_debt:
                # self._repay_debt()
                self._pay_interest()
                self._bankruptcy_check()
            self._inflation()
            # self._nominal_interest_rate()
            # Store values
            self.production_hist[:, i] = self.production
            self.debt_hist[:, i] = self.debt
            self.money_hist[:, i] = self.money
            self.inflation_rate_hist[i] = self.inflation_rate
        
        # if (self.production_hist < 1).any():
        #     print("There were production values below 1")
        #     idx_below_1 = np.where(self.production_hist < 1)
        #     idx_comp = idx_below_1[0]
        #     idx_time = idx_below_1[1]
        #     print("All")
        #     print(idx_below_1)
        #     print("Time values:")
        #     print(idx_time)
            
        #     # Print the value of the first company at the first time going below 1
        #     idx_first_below_1 = (idx_comp[0], idx_time[0])
        #     print("First to go below 1:")
        #     print(self.production_hist[idx_first_below_1])
            
            
        
        # Save data to file
        self._data_to_file()
    
    
    def parameter_change_simulation(self, func_buyer_seller_idx, r_vals: np.ndarray, N_repeats: int) -> None:
        """Run simulation for different interest values r, each value is run N_repeats times. Store in file.

        Args:
            func_buyer_seller_idx (function): Choice of buyer and seller idx e.g. well mixed or 1d neighbours.
            r_vals (ndarraylike): Interest values.
            N_repeats (int): Times each r value is repeated.
        """
        # Store original interest value so can change back afterwards
        r_original = self.real_interest_rate
        
        # Empty lists for data storage
        data_production = np.zeros((len(r_vals) * N_repeats, self.time_steps))
        data_debt = np.zeros_like(data_production)
        data_money = np.zeros_like(data_production)
        
        # Get the data
        idx_counter = 0
        for r in tqdm(r_vals):
            self.real_interest_rate = r  # Update the class' interest value
            for _ in range(N_repeats):
                # Generate data, take mean over companies, then store in data arrays
                self.simulation(func_buyer_seller_idx)
                production_mean = np.mean(self.production_hist, axis=0)
                debt_mean = np.mean(self.debt_hist, axis=0)
                money_mean = np.mean(self.money_hist, axis=0)
                
                data_production[idx_counter, :] = production_mean
                data_debt[idx_counter, :] = debt_mean
                data_money[idx_counter, :] = money_mean
                
                idx_counter += 1

        # Save data to file. Combine data into one array, then save.
        # Want to save r_vals as well, so expand r_vals to fit in
        r_vals_save = np.empty_like(data_production)
        r_vals_save[:, 0] = np.repeat(r_vals, N_repeats)
        
        all_data = np.empty((len(data_production), self.time_steps, 4))
        all_data[:, :, 0] = np.array(data_production)
        all_data[:, :, 1] = np.array(data_debt)
        all_data[:, :, 2] = np.array(data_money)
        all_data[:, :, 3] = r_vals_save

        filename = Path.joinpath(self.dir_path_output, "parameter_change_" + self.file_parameter_addon + ".npy")
        np.save(filename, arr=all_data)

        # Restore orignal interest value
        self.real_interest_rate = r_original
        

# Parameters
N_agents = 100
time_steps = 1250
real_interest_rate = 0.5  # gamma
money_to_production_efficiency = 1 #np.round(1.8 * real_interest_rate, 3)  # alpha, growth exponent
equilibrium_distance_fraction = 5e-2  # epsilon
include_debt = True
buy_fraction = 1  # sigma

# For parameter_change_simulation
interest_values = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
N_repeats = 10


if __name__ == "__main__":
    print("You ran the wrong script :)")