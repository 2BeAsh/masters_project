import numpy as np
from tqdm import tqdm


class DebtDeflation():
    def __init__(self, number_of_companies: int, money_to_production_efficiency: float, interest_rate: float, buy_fraction: float, equilibrium_distance_fraction: float, include_debt: bool, time_steps: int):
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
        self.r = interest_rate  # Interest rate
        self.buy_fraction = buy_fraction  # When doing a transaction, how large a fraction of min(seller production, buyer money) is used.
        self.epsilon = equilibrium_distance_fraction  # In inflation updates, the fraction the system goes toward equilibrlium.
        self.include_debt = include_debt
        self.time_steps = time_steps  # Steps taken in system evolution.
        
        # Local paths for saving files.
        self.dir_path = "code/"
        self.dir_path_output = self.dir_path + "output/"
        self.dir_path_image = self.dir_path + "image/"
        self.file_parameter_addon = f"Steps{self.time_steps}_Companies{self.N}_Interest{self.r}_Efficiency{self.alpha}_BuyFraction{self.buy_fraction}_EquilibriumStep{self.epsilon}_wellmixed"


    def _initial_market(self) -> None:
        """Initialize market.
        Production = 1, debt = 0, money = 1
        """
        self.production = np.ones(self.N)
        self.debt = np.zeros(self.N)
        self.money = np.ones(self.N)
    
        
    def _step(self, buyer_idx, seller_idx) -> None:
        """First check if the buyer needs to take a loan to match the sellers production, then make transaction and update values

        Args:
            buyer_idx (int): _description_
            seller_idx (int): _description_
        """
        # If the buyer's money is less than the seller's production, 
        # the buyer takes a loan to try and match the production. 
        # The buyer cannot take a loan larger than its company size
        if self.money[buyer_idx] < self.production[seller_idx] and self.include_debt:
            # Calculate loan size
            production_money_difference = self.production[seller_idx] - self.money[buyer_idx]
            production_debt_difference = self.production[buyer_idx] - self.r * self.debt[buyer_idx]  # Production is ideal money gained, r * d is money lost. 
            loan_size = np.min([production_money_difference, production_debt_difference])
            # Update values
            self.money[buyer_idx] += loan_size
            self.debt[buyer_idx] += loan_size
        
        amount_bought = self.buy_fraction * np.min([self.production[seller_idx], self.money[buyer_idx]])
        # Update values
        self.money[seller_idx] += amount_bought
        self.money[buyer_idx] -= amount_bought
        self.production[buyer_idx] += self.alpha * amount_bought
    
    
    def _buyer_seller_idx_money_scaling(self) -> tuple:
        """Choosing the seller and buyer indices. 
        The seller is uniformly drawn. The buyer is drawn proportional to its money.

        Returns:
            tuple: buyer idx, seller idx
        """
        seller_idx = np.random.randint(low=0, high=self.N)
        money_wrt_negative_vals = np.maximum(self.money, 0)  # If has negative money, give a 0 chance to buy.
        buyer_prob = money_wrt_negative_vals / money_wrt_negative_vals.sum()  # The probability to buy is proportional to ones money. 
        buyer_idx = np.random.choice(np.arange(self.N), p=buyer_prob)
        return buyer_idx, seller_idx
    
    
    def _buyer_seller_idx_uniform(self) -> tuple:
        """Choosing the seller and buyer indices.
        Both seller and buyer are uniformly drawn.
        
        Returns:
            tuple: buyer idx, seller idx
        """
        seller_idx = np.random.randint(low=0, high=self.N)
        buyer_idx = np.random.randint(low=0, high=self.N)
        return buyer_idx, seller_idx
        
    
    def _pay_rent(self):
        """Money is update according to how much debt the firm holds.
        """
        self.money -= self.r * self.debt
    
    
    def _bankruptcy_check(self):
        """Bankrupt if p + m < d. If goes bankrupt, start a new company in its place with initial values.
        Intuition is that when m=0, you must have high enough production to pay off the debt in one transaction i.e. p > d"""
        # Find all companies who have gone bankrupt
        bankrupt_idx = np.where(self.production + self.money < self.debt * self.r)
        
        # Print one of the bankrupt comapny's values
        idx1 = bankrupt_idx[0]
        print(f"Production: \n{self.production[idx1]}, \nMoney: \n{self.money[idx1]}, \nDebt: \n{self.debt[idx1]}")
        print("")
        print("")
        
        # Set their values to the initial values
        self.money[bankrupt_idx] = 1
        self.debt[bankrupt_idx] = 0
        self.production[bankrupt_idx] = 1


    def _inflation(self):
        """Update the money according to the inflation rule:
        mi <- mi + epsilon * (P / M - 1) * mi

        Args:
            epsilon (float): How much of the distance to the equilibrium the step covers.
        """
        # Find distance to equilibrium
        suply = self.production.sum()
        demand = self.money.sum()
        distance_to_equilibrium = (suply / demand - 1) * self.money
        # Perform the update
        self.money = self.money + self.epsilon * distance_to_equilibrium


    def _data_to_file(self) -> None:
        # Collect data to store in one array
        all_data = np.empty((self.N, self.time_steps, 3))
        all_data[:, :, 0] = self.production_hist
        all_data[:, :, 1] = self.debt_hist
        all_data[:, :, 2] = self.money_hist

        filename = self.dir_path_output + self.file_parameter_addon + ".npy"
        np.save(filename, arr=all_data)
    

    def simulation(self) -> None:
        # Initialize market
        self._initial_market()
        # History and its first value
        self.production_hist = np.zeros((self.N, self.time_steps))
        self.debt_hist = np.zeros_like(self.production_hist)
        self.money_hist = np.zeros_like(self.production_hist)
        self.production_hist[:, 0] = self.production
        self.debt_hist[:, 0] = self.debt
        self.money_hist[:, 0] = self.money
        
        # Time evolution
        for i in tqdm(range(1, self.time_steps)):
            # Make N transactions
            for _ in range(self.N):
                # Pick buyer and seller
                buyer_idx, seller_idx = self._buyer_seller_idx_money_scaling()
                self._step(buyer_idx, seller_idx)
            # Pay rent and check for bankruptcy
            if self.include_debt:
                self._pay_rent()
                self._bankruptcy_check()
            self._inflation()
            # Store values
            self.production_hist[:, i] = self.production
            self.debt_hist[:, i] = self.debt
            self.money_hist[:, i] = self.money
        
        # Save data to file
        self._data_to_file()
    

# Parameters
N_agents = 100
time_steps = 1000
interest = 1
money_to_production_efficiency = 0.05  # alpha, growth exponent
buy_fraction = 1  # sigma
equilibrium_distance_fraction = 0.01  # epsilon

debtdeflation = DebtDeflation(number_of_companies=N_agents, 
                                money_to_production_efficiency=money_to_production_efficiency, 
                                interest_rate=interest, 
                                buy_fraction=buy_fraction, 
                                equilibrium_distance_fraction=equilibrium_distance_fraction, 
                                time_steps=time_steps,
                                include_debt=True)

filename_parameter_addon = debtdeflation.file_parameter_addon

if __name__ == "__main__":
    debtdeflation.simulation()
