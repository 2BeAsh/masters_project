import numpy as np
from tqdm import tqdm
import inspect


class DebtDeflation():
    def __init__(self, number_of_companies: int, money_to_production_efficiency: float, loan_probability: float, interest_rate: float, buy_fraction: float, equilibrium_distance_fraction: float, time_steps: int):
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
        self.beta = loan_probability  # Probability to take a loan
        self.r = interest_rate  # Interest rate
        self.buy_fraction = buy_fraction  # When doing a transaction, how large a fraction of min(seller production, buyer money) is used.
        self.epsilon = equilibrium_distance_fraction  # In inflation updates, the fraction the system goes toward equilibrlium.
        self.time_steps = time_steps  # Steps taken in system evolution.
        
        # Local paths for saving files.
        self.dir_path = "code/"
        self.dir_path_output = self.dir_path + "output/"
        self.dir_path_image = self.dir_path + "image/"
        self.file_parameter_addon = f"Steps{self.time_steps}_Companies{self.N}_Interest{self.r}_Efficiency{self.alpha}_LoanProb{self.beta}_BuyFraction{self.buy_fraction}_EquilibriumStep{self.epsilon}"


    def _initial_market(self) -> None:
        """Initialize market.
        Production = 1, debt = 0, money = 1
        """
        self.production = np.ones(self.N)
        self.debt = np.zeros(self.N)
        self.money = np.ones(self.N)
    
    
    def _step(self, buyer_idx, seller_idx) -> None:
        """First make transaction, then check if the buyer takes a loan.

        Args:
            buyer_idx (int): _description_
            seller_idx (int): _description_
        """
        # Amount bought is limited by the sellers production and the buyers money
        amount_bought = self.buy_fraction * np.min([self.production[seller_idx], self.money[buyer_idx]])
        
        self.money[seller_idx] += amount_bought
        self.money[buyer_idx] -= amount_bought
        self.production[buyer_idx] += self.alpha * amount_bought
        
        # Buyer takes loan with probability beta, increasing debt and money by the same amount
        if self.beta > np.random.uniform(low=0, high=1):
            self.debt[buyer_idx] += amount_bought / self.r
            self.money[buyer_idx] += amount_bought / self.r
    
    
    def _buyer_seller_idx_money_scaling(self) -> tuple:
        """Choosing the seller and buyer indices. 
        The seller is uniformly drawn. The buyer is drawn proportional to its money.

        Returns:
            tuple: buyer idx, seller idx
        """
        seller_idx = np.random.randint(low=0, high=self.N)
        buyer_prob = self.money / self.money.sum()  # The probability to buy is proportional to ones money
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
        """Companies with negative money goes bankrupt, starting a new company in its place with initial values"""
        # Find all companies who have gone bankrupt
        bankrupt_idx = np.where(self.money < 1e-10)
        # Set their values to the initial values
        self.money[bankrupt_idx] = 1
        self.debt[bankrupt_idx] = 0
        self.production[bankrupt_idx] = 1


    def _inflation(self, epsilon):
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
        self.money = self.money + epsilon * distance_to_equilibrium


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
                buyer_idx, seller_idx = self._buyer_seller_idx_uniform()
                self._step(buyer_idx, seller_idx)
            # Pay rent and check for bankruptcy
            # self._pay_rent()
            # self._bankruptcy_check()
            self._inflation(self.epsilon)
            # Store values
            self.production_hist[:, i] = self.production
            self.debt_hist[:, i] = self.debt
            self.money_hist[:, i] = self.money
        
        # Save data to file
        self._data_to_file()
        

if __name__ == "__main__":
    # Parameters
    N_agents = 100
    time_steps = 1000
    interest = 1
    money_to_production_efficiency = 0.05  # alpha
    loan_probability = 0.0  # beta
    buy_fraction = 0.9  # sigma
    equilibrium_distance_fraction = 1 / 10  # epsilon
    
    debtdeflation = DebtDeflation(number_of_companies=N_agents, 
                                  money_to_production_efficiency=money_to_production_efficiency, 
                                  loan_probability=loan_probability, 
                                  interest_rate=interest, 
                                  buy_fraction=buy_fraction, 
                                  equilibrium_distance_fraction=equilibrium_distance_fraction, 
                                  time_steps=time_steps)
    
    debtdeflation.simulation()
