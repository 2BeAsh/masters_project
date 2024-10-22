import numpy as np
import networkx as nx
from bank_master import BankDebtDeflation, time_steps, N_companies, alpha, interest_rate_change_size, beta_mutation_size, beta_update_method, interest_update_method
from pathlib import Path


class BankNetwork(BankDebtDeflation):
    def __init__(self, number_of_companies: int, money_to_production_efficiency: float, interest_rate_change_size: float, beta_mutation_size: float, 
                 beta_update_method: str, interest_update_method: int, p_edge: float, time_steps: int):
        """_summary_

        Args:
            number_of_companies (int): _description_
            money_to_production_efficiency (float): _description_
            time_steps (int): _description_
        """
        # Get master methods
        super().__init__(number_of_companies, money_to_production_efficiency, interest_rate_change_size, beta_mutation_size, beta_update_method, interest_update_method, time_steps)

        # New variables
        self.p_edge = p_edge  # Chance for two nodes to be connected

        # Local paths for saving files.
        self.dir_path_image = Path.joinpath(self.dir_path, "image_bank", "network")
        self.file_parameter_addon = self.file_parameter_addon_base + f"pedge{p_edge}" + "_network"
        
        
    def _initialize_network_ER(self):
        self.network = nx.erdos_renyi_graph(n=self.N, p=self.p_edge)
            
    
    def _buyer_seller_idx(self) -> tuple:
        """Pick buyer randomly, pick seller as one of its neighbours
        """
        # Buyer is random
        buyer_idx = np.random.randint(low=0, high=self.N)
        
        # Seller is one of buyer's neighbours
        # Get neighbouring nodes and pick one randomly
        neighbours = list(self.network.neighbors(buyer_idx))
        seller_idx = np.random.choice(a=neighbours)

        return buyer_idx, seller_idx
        
    
    # Overwrite bankruptcy method, as need to update the network edges
    # def _company_bankruptcy_check(self):
    #     pass


    def run_simulation(self):
        self._initialize_network_ER()
        func_idx = self._buyer_seller_idx
        self.simulation(func_idx)
    

# Parameter values
p_edge = 0.1
        
bank_network = BankNetwork(number_of_companies=N_companies, 
                               money_to_production_efficiency=alpha,
                               interest_rate_change_size=interest_rate_change_size, 
                               beta_mutation_size=beta_mutation_size,
                               beta_update_method=beta_update_method,
                               interest_update_method=interest_update_method,
                               p_edge=p_edge,
                               time_steps=time_steps)


filename_parameter_addon = bank_network.file_parameter_addon

if __name__ == "__main__":
    bank_network.run_simulation()