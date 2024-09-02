import numpy as np
from tqdm import tqdm
from debt_deflation_master import DebtDeflation, N_agents, time_steps, real_interest_rate, money_to_production_efficiency, buy_fraction, equilibrium_distance_fraction, include_debt, interest_values, N_repeats
from pathlib import Path


class DebtDeflationWellMixed(DebtDeflation):
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
        # Get master methods
        super().__init__(number_of_companies, money_to_production_efficiency, real_interest_rate, buy_fraction, equilibrium_distance_fraction, include_debt, time_steps)
        
        # Local paths for saving files.
        self.dir_path_image = Path.joinpath(self.dir_path, "image", "wellmixed")
        self.file_parameter_addon = self.file_parameter_addon_base + "_wellmixed"

    
    def _buyer_seller_idx_uniform(self) -> tuple:
        """Choosing the seller and buyer indices.
        Both seller and buyer are uniformly drawn.
        
        Returns:
            tuple: buyer idx, seller idx
        """
        seller_idx = np.random.randint(low=0, high=self.N)
        buyer_idx = np.random.randint(low=0, high=self.N)
        return buyer_idx, seller_idx
    
    
    def _buyer_seller_idx_money_scaling(self) -> tuple:
        """Choosing the seller and buyer indices. 
        The seller is uniformly drawn. The buyer is drawn proportional to its money.

        Returns:
            tuple: buyer idx, seller idx
        """
        seller_idx = np.random.randint(low=0, high=self.N)
        money_wrt_negative_vals = np.maximum(self.money, 0)  # If has negative money, give a 0 chance to buy.
        money_wrt_negative_vals_sum = money_wrt_negative_vals.sum()
        buyer_prob = money_wrt_negative_vals / money_wrt_negative_vals_sum   # The probability to buy is proportional to ones money. 
        
        # If all companies have negative money, the probability is instead based on distance from poorest company
        if money_wrt_negative_vals_sum < 1e-5:  # Small number, effectively 0
            lowest_money = np.min(self.money)    
            dist_to_lowest_money = self.money - lowest_money
            buyer_prob = dist_to_lowest_money / dist_to_lowest_money.sum()  # Normalize to get probablility
            
        buyer_idx = np.random.choice(np.arange(self.N), p=buyer_prob)
        return buyer_idx, seller_idx
        
        
    def run_simulation(self):
        # self.simulation(self._buyer_seller_idx_money_scaling)
        self.simulation(self._buyer_seller_idx_uniform)
        
        
    def run_parameter_change_simulation(self, r_vals, N_repeats):
        self.parameter_change_simulation(self._buyer_seller_idx_money_scaling, r_vals, N_repeats)
    

# Parameters
debtdeflation_wellmixed = DebtDeflationWellMixed(number_of_companies=N_agents, 
                                money_to_production_efficiency=money_to_production_efficiency, 
                                real_interest_rate=real_interest_rate, 
                                buy_fraction=buy_fraction, 
                                equilibrium_distance_fraction=equilibrium_distance_fraction, 
                                time_steps=time_steps,
                                include_debt=include_debt)

filename_parameter_addon = debtdeflation_wellmixed.file_parameter_addon

if __name__ == "__main__":
    # debtdeflation_wellmixed.run_simulation()
    debtdeflation_wellmixed.run_parameter_change_simulation(interest_values, N_repeats)
    