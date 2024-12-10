import numpy as np
from tqdm import tqdm
from pathlib import Path
from redistribution_master import Workforce, number_of_companies, number_of_workers, salary_increase, machine_cost, time_steps


class BankWellMixed(Workforce):
    def __init__(self, number_of_companies, number_of_workers, salary_factor, machine_cost, time_steps):
        """_summary_

        Args:
            number_of_companies (int): _description_
            money_to_production_efficiency (float): _description_
            time_steps (int): _description_
        """
        # Get master methods
        super().__init__(number_of_companies, number_of_workers, salary_factor, machine_cost, time_steps)

    
    def run_store_values(self):
        self.store_values()
        
        
    def run_store_peak_rho_space(self):
        self.store_peak_rho_space()
        
        
bank_wellmixed = BankWellMixed(number_of_companies, number_of_workers, salary_increase, machine_cost, time_steps)

if __name__ == "__main__":
    # bank_wellmixed.run_store_values()
    bank_wellmixed.run_store_peak_rho_space()
    print("Stored Values")