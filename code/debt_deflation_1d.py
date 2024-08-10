import numpy as np
from tqdm import tqdm
from debt_deflation_master import DebtDeflation, N_agents, time_steps, interest, money_to_production_efficiency, buy_fraction, equilibrium_distance_fraction, include_debt


class DebtDeflation1d(DebtDeflation):
    def __init__(self, number_of_companies: int, money_to_production_efficiency: float, 
                 interest_rate: float, buy_fraction: float, equilibrium_distance_fraction: float, 
                 neighbour_width: int, include_debt: bool, time_steps: int):
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
        super().__init__(number_of_companies, money_to_production_efficiency, interest_rate, buy_fraction, equilibrium_distance_fraction, include_debt, time_steps)
        
        # New parameters
        self.nbor = neighbour_width
        
        # Updated image path
        self.dir_path_image = self.dir_path + "image/" + "1d/"
        self.file_parameter_addon = self.file_parameter_addon_base +  f"_nborwidth{self.nbor}_1D"
    
    
    def _seller_idx(self, buyer_idx) -> int:
        nbor_dist = np.random.randint(-self.nbor, self.nbor + 1)
        seller_idx = buyer_idx + nbor_dist
        if seller_idx >= self.N:  # Right boundary
            seller_idx = buyer_idx - 1
        elif seller_idx < 0:  # Left boundary
            seller_idx = buyer_idx + 1

        return seller_idx
    
    
    def _buyer_seller_idx_money_scaling_1d(self) -> tuple:
        """Pick buyer based proportional to money, pick seller as one of its neighbours
        """
        # Buyer is prop to money
        money_wrt_negative_vals = np.maximum(self.money, 0)  # If has negative money, give a 0 chance to buy.
        buyer_prob = money_wrt_negative_vals / money_wrt_negative_vals.sum()  # The probability to buy is proportional to one's money. 
        buyer_idx = np.random.choice(np.arange(self.N), p=buyer_prob)
        # Seller
        seller_idx = self._seller_idx(buyer_idx)
        
        return buyer_idx, seller_idx
        

    def _buyer_seller_idx_uniform_1d(self) -> tuple:
        """Pick buyer randomly, pick seller as one of its neighbours
        """
        # Buyer is prop to money
        buyer_idx = np.random.randint(low=0, high=self.N)
        
        # Seller
        seller_idx = self._seller_idx(buyer_idx)
        
        return buyer_idx, seller_idx


    def run_simulation(self):
        self.simulation(self._buyer_seller_idx_uniform_1d)

# Parameters unique to 1d i.e. not used in well mixed
neighbour_width = 1


# Create instance of class 
debtdeflation_1d = DebtDeflation1d(number_of_companies=N_agents, 
                            money_to_production_efficiency=money_to_production_efficiency, 
                            interest_rate=interest, 
                            buy_fraction=buy_fraction, 
                            equilibrium_distance_fraction=equilibrium_distance_fraction, 
                            neighbour_width=neighbour_width,
                            time_steps=time_steps,
                            include_debt=include_debt)

# Get filename path for plotting
filename_parameter_addon_1d = debtdeflation_1d.file_parameter_addon


if __name__ == "__main__":
    # Run the simulation
    debtdeflation_1d.run_simulation()