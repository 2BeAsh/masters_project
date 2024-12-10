import numpy as np
from tqdm import tqdm
from pathlib import Path
from workforce_master import Workforce, number_of_companies, number_of_workers, interest_rate_change_size, salary_increase, time_steps


class BankWellMixed(Workforce):
    def __init__(self, number_of_companies, number_of_workers, interest_rate_change_size, salary_factor, time_steps):
        """_summary_

        Args:
            number_of_companies (int): _description_
            money_to_production_efficiency (float): _description_
            time_steps (int): _description_
        """
        # Get master methods
        super().__init__(number_of_companies, number_of_workers, interest_rate_change_size, salary_factor, time_steps)

    
    def run_store_values(self):
        self.store_values()
        
        
bank_wellmixed = BankWellMixed(number_of_companies, number_of_workers, interest_rate_change_size, salary_increase, time_steps)

if __name__ == "__main__":
    bank_wellmixed.run_store_values()