import numpy as np
from tqdm import tqdm
from bank_master import BankDebtDeflation, time_steps, N_companies, alpha, interest_rate_change_size, beta_mutation_size, beta_update_method, derivative_order
from pathlib import Path


class Bank1d(BankDebtDeflation):
    def __init__(self, number_of_companies: int, money_to_production_efficiency: float, interest_rate_change_size: float, beta_mutation_size: float, beta_update_method: str, derivative_order: int, time_steps: int):
        """_summary_

        Args:
            number_of_companies (int): _description_
            money_to_production_efficiency (float): _description_
            time_steps (int): _description_
        """
        # Get master methods
        super().__init__(number_of_companies, money_to_production_efficiency, interest_rate_change_size, beta_mutation_size, beta_update_method, derivative_order, time_steps)

        # Local paths for saving files.
        self.dir_path_image = Path.joinpath(self.dir_path, "image_bank", "1d")
        self.file_parameter_addon = self.file_parameter_addon_base + "_1d"
        
        
    def _seller_idx(self, buyer_idx) -> int:
        nbor_dist = np.random.randint(low=-1, high=2)  # High not included
        seller_idx = buyer_idx + nbor_dist
        seller_idx = np.clip(a=seller_idx, a_min=0, a_max=self.N-1)
        return seller_idx
    
    
    def _buyer_seller_idx_uniform(self) -> tuple:
        """Pick buyer randomly, pick seller as one of its neighbours
        """
        # Buyer is random
        buyer_idx = np.random.randint(low=0, high=self.N)
        
        # Seller is one of buyer's neighbours
        seller_idx = self._seller_idx(buyer_idx)
        
        return buyer_idx, seller_idx


    def run_simulation(self):
        func_idx = self._buyer_seller_idx_uniform
        self.simulation(func_idx)
        
        
bank_1d = Bank1d(number_of_companies=N_companies, 
                               money_to_production_efficiency=alpha,
                               interest_rate_change_size=interest_rate_change_size, 
                               beta_mutation_size=beta_mutation_size,
                               beta_update_method=beta_update_method,
                               derivative_order=derivative_order,
                               time_steps=time_steps)


filename_parameter_addon = bank_1d.file_parameter_addon

if __name__ == "__main__":
    bank_1d.run_simulation()