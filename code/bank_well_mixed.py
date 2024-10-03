import numpy as np
from tqdm import tqdm
from bank_master import BankDebtDeflation, time_steps, N_companies, alpha, interest_rate_change_size, beta_mutation_size, beta_update_method, interest_update_method
from pathlib import Path


class BankWellMixed(BankDebtDeflation):
    def __init__(self, number_of_companies: int, money_to_production_efficiency: float, interest_rate_change_size: float, beta_mutation_size: float, beta_update_method: str, interest_update_method, time_steps: int):
        """_summary_

        Args:
            number_of_companies (int): _description_
            money_to_production_efficiency (float): _description_
            bank_greed (float): _description_
            time_steps (int): _description_
        """
        # Get master methods
        super().__init__(number_of_companies, money_to_production_efficiency, interest_rate_change_size, beta_mutation_size, beta_update_method, interest_update_method, time_steps)

        # Local paths for saving files.
        self.dir_path_image = Path.joinpath(self.dir_path, "image_bank", "wellmixed")
        self.file_parameter_addon = self.file_parameter_addon_base + "_wellmixed"
        
        
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


    def run_simulation(self, idx_choice="uniform"):
        assert idx_choice in ["uniform"]
        if idx_choice == "uniform":
            func_idx = self._buyer_seller_idx_uniform
        self.simulation(func_idx)
        
        
bank_wellmixed = BankWellMixed(number_of_companies=N_companies, 
                               money_to_production_efficiency=alpha,
                               interest_rate_change_size=interest_rate_change_size, 
                               beta_mutation_size=beta_mutation_size,
                               beta_update_method=beta_update_method,
                               derivative_order=interest_update_method,
                               time_steps=time_steps)


filename_parameter_addon = bank_wellmixed.file_parameter_addon

if __name__ == "__main__":
    bank_wellmixed.run_simulation(idx_choice="uniform")