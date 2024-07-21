import numpy as np
from tqdm import tqdm
from debt_deflation_well_mixed import DebtDeflation, N_agents, time_steps, interest, money_to_production_efficiency, buy_fraction, equilibrium_distance_fraction


class DebtDeflation_1d(DebtDeflation):
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
        super().__init__(number_of_companies, money_to_production_efficiency, interest_rate, buy_fraction, equilibrium_distance_fraction, include_debt, time_steps)
        self.file_parameter_addon = f"Steps{self.time_steps}_Companies{self.N}_Interest{self.r}_Efficiency{self.alpha}_BuyFraction{self.buy_fraction}_EquilibriumStep{self.epsilon}_1D"
    
    
    def _buyer_seller_idx_1d(self) -> tuple:
        """Pick buyer based proportional to money, pick seller as one of its neighbours
        """
        # Buyer is prop to money
        money_wrt_negative_vals = np.maximum(self.money, 0)  # If has negative money, give a 0 chance to buy.
        buyer_prob = money_wrt_negative_vals / money_wrt_negative_vals.sum()  # The probability to buy is proportional to one's money. 
        buyer_idx = np.random.choice(np.arange(self.N), p=buyer_prob)
        
        # Seller one of its neighbours, taking closed boundaries into account
        minus_or_plus_one = 2 * np.random.randint(low=0, high=2) - 1
        seller_idx = buyer_idx + minus_or_plus_one
        if seller_idx == self.N:  # Right boundary
            seller_idx = buyer_idx - 1
        elif seller_idx < 0:  # Left boundary
            seller_idx = buyer_idx + 1
        
        return buyer_idx, seller_idx
        
        
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
                buyer_idx, seller_idx = self._buyer_seller_idx_1d()
                self._step(buyer_idx, seller_idx)
            # Pay rent and check for bankruptcy
            if self.include_debt:
                self._pay_rent()
                self._bankruptcy_check()
            self._inflation(self.epsilon)
            # Store values
            self.production_hist[:, i] = self.production
            self.debt_hist[:, i] = self.debt
            self.money_hist[:, i] = self.money
        
        # Save data to file
        self._data_to_file()



debtdeflation_1d = DebtDeflation_1d(number_of_companies=N_agents, 
                            money_to_production_efficiency=money_to_production_efficiency, 
                            interest_rate=interest, 
                            buy_fraction=buy_fraction, 
                            equilibrium_distance_fraction=equilibrium_distance_fraction, 
                            time_steps=time_steps,
                            include_debt=True)
filename_parameter_addon_1d = debtdeflation_1d.file_parameter_addon


if __name__ == "__main__":
    debtdeflation_1d.simulation()