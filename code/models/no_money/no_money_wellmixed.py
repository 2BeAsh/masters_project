import numpy as np
from tqdm import tqdm
from pathlib import Path
from no_money_master import BankNoMoney, time_steps, number_of_companies, time_steps, money_to_production_efficiency, interest_rate_change_size, beta_mutation_size, beta_update_method, interest_update_method


class BankWellMixed(BankNoMoney):
    def __init__(self, number_of_companies: int, money_to_production_efficiency: float, interest_rate_change_size: float, beta_mutation_size: float, beta_update_method: str, interest_update_method: str, time_steps: int):
        """_summary_

        Args:
            number_of_companies (int): _description_
            money_to_production_efficiency (float): _description_
            bank_greed (float): _description_
            time_steps (int): _description_
        """
        # Get master methods
        super().__init__(number_of_companies, money_to_production_efficiency, interest_rate_change_size, beta_mutation_size, beta_update_method, interest_update_method, time_steps)
        
        
    def _buyer_seller_idx_uniform(self) -> tuple:
        """Choosing the seller and buyer indices.
        Both seller and buyer are uniformly drawn. 
        First draw seller, then drawn buyer from the remaining options.
        
        Returns:
            tuple: buyer idx, seller idx
        """
        
        seller_idx = np.random.randint(low=0, high=self.N)
        available_buyer_idx = np.arange(self.N)[np.arange(self.N) != seller_idx]
        buyer_idx = np.random.choice(a=available_buyer_idx)
        return buyer_idx, seller_idx


    def run_store_values(self):
        func_idx = self._buyer_seller_idx_uniform
        self.store_values(func_idx)
        
        
bank_wellmixed = BankWellMixed(number_of_companies=number_of_companies, 
                               money_to_production_efficiency=money_to_production_efficiency,
                               interest_rate_change_size=interest_rate_change_size, 
                               beta_mutation_size=beta_mutation_size,
                               beta_update_method=beta_update_method,
                               interest_update_method=interest_update_method,
                               time_steps=time_steps)

if __name__ == "__main__":
    bank_wellmixed.run_store_values()    