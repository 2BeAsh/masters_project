import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Callable


class BankDebtDeflation():
    def __init__(self, number_of_companies: int, money_to_production_efficiency: float, interest_rate_change_size=0.01, 
                 beta_mutation_size=0.05, beta_update_method="production", derivative_order=2, time_steps=1000):
        """Initializer. Children must define the following:
            self.file_parameter_addon

        Args:
            number_of_companies (int): _description_
            money_to_production_efficiency (float): _description_
            time_steps (int): _description_
        """
        # Check different input values are correct
        assert beta_update_method in ["production", "random"]
        assert derivative_order in [1, 2]
        
        # Set parameters
        self.N = number_of_companies
        self.alpha = money_to_production_efficiency  # Money to production efficiency
        self.interest_rate_change_size = interest_rate_change_size
        self.beta_mutation_size = beta_mutation_size
        self.beta_update_method = beta_update_method  # How new beta values are chosen
        self.derivative_order = derivative_order  # Order of derivative used for interest rate adjustment
        self.time_steps = time_steps  # Steps taken in system evolution.

        # Local paths for saving files.
        file_path = Path(__file__)
        self.dir_path = file_path.parent
        self.dir_path_output = Path.joinpath(self.dir_path, "output")
        self.dir_path_image = Path.joinpath(self.dir_path, "image_bank")
        self.file_parameter_addon_base = f"Steps{self.time_steps}_N{self.N}_alpha{self.alpha}_dr{interest_rate_change_size}_dbeta{beta_mutation_size}_betaUpdate{beta_update_method}_deriv{derivative_order}"

        np.random.seed(1)  # For variable control
        
        # Parameters values are initialized in _initialize_market


    def _initialize_market(self):
        self.p = np.ones(self.N) * 1.  # * 1. converts to float
        self.d = np.zeros(self.N) * 1.
        self.m = np.ones(self.N) * 1.
        self.beta = np.random.uniform(low=0, high=1, size=self.N)  # Risk factor in loans
        self.bank_money = 1  # Value of bank_money doesn't matter for derivatives 
        self.interest_rate = 0.1  # OBS might choose another value.
        self.interest_rate_PD_adjusted = self.interest_rate  # Assume no chance of bankruptcy at start
        self.interest_action = 1  # -1 means reduce, 1 means increase. Arbitrarly set to increase at start.
        self.did_not_take_loan = 0


    def _initialize_hist_arrays(self):
        """Create empty arrays to be filled with data during simulation, and fill in the initial value.
        """
        # Create arrays for storage of values
        self.p_hist = np.empty((self.N, self.time_steps))
        self.d_hist = np.empty_like(self.p_hist)
        self.m_hist = np.empty_like(self.p_hist)
        self.beta_hist = np.empty_like(self.p_hist)
        self.bank_fortune_hist = np.empty(self.time_steps)
        self.interest_rate_hist = np.empty_like(self.bank_fortune_hist)
        self.interest_rate_PD_adjusted_hist = np.empty_like(self.bank_fortune_hist)
        self.did_not_take_loan_hist = np.empty(self.time_steps)
        self.N_bankrupt_hist = np.empty(self.time_steps)
        self.first_derivative_hist = np.empty(self.time_steps)
        self.second_derivative_hist = np.empty(self.time_steps)

        # Set initial values
        self.p_hist[:, 0] = self.p * 1
        self.d_hist[:, 0] = self.d * 1
        self.m_hist[:, 0] = self.m * 1
        self.beta_hist[:, 0] = self.beta * 1
        self.bank_fortune_hist[0] = self.bank_money * 1  # No debt initially
        self.interest_rate_hist[0] = self.interest_rate * 1
        self.interest_rate_PD_adjusted_hist[0] = self.interest_rate_PD_adjusted * 1
        self.did_not_take_loan_hist[0] = self.did_not_take_loan * 1  # Must have initial value.
        self.derivative_1st = 1  # First 5 steps no value is calculated, so set to 1
        self.derivative_2nd = 1


    def _take_loan(self, buyer_idx, seller_idx):
        # Establish loan boundaries and clip the smart loan to match the boundaries
        debt_max = self.p[seller_idx] - self.m[buyer_idx]  # No reason to go beyond the seller's production
        debt_min = 0  # Cannot take negative debt
        delta_debt = (self.beta[buyer_idx] * self.p[buyer_idx] + self.m[buyer_idx]) / self.interest_rate_PD_adjusted - self.d[buyer_idx]  # OBS has removed money from debt term
        delta_debt_clipped = np.clip(a=delta_debt, a_min=debt_min, a_max=debt_max)

        # If debt == 0 (could be because non-clipped debt was negative), no loan was taken
        if delta_debt_clipped == 0:
            self.did_not_take_loan += 1
        # Update values
        self.m[buyer_idx] += delta_debt_clipped
        self.d[buyer_idx] += delta_debt_clipped
        self.bank_money -= delta_debt_clipped


    def _transaction(self, buyer_idx, seller_idx):
        # Take loan if necessary
        if self.p[seller_idx] > self.m[buyer_idx]:
            self._take_loan(buyer_idx, seller_idx)
        else:
            self.did_not_take_loan += 1
        # Amount transferred/Bought
        delta_money = np.min((self.p[seller_idx], self.m[buyer_idx]))
        delta_money = np.max((delta_money, 0))  # Disallow negative values
        self.m[buyer_idx] -= delta_money
        self.m[seller_idx] += delta_money
        self.p[buyer_idx] += delta_money * self.alpha


    def _negative_money_to_loan(self):
        """If any company has entered negative money, take a loan to correct it.
        """
        # Find companies with negative money
        idx_negative_money = np.where(self.m < 0)
        # Take debt to set money to zero
        debt = np.abs(self.m[idx_negative_money]) + 0.1  # 0.1 is negligeable, but makes log-plotting prettier
        # Update company money and debt values, and bank money
        self.m[idx_negative_money] += debt
        self.d[idx_negative_money] += debt
        self.bank_money -= debt.sum()


    def _pay_interest(self):
        """Pay interest to the bank.
        """
        self.m -= self.interest_rate_PD_adjusted * self.d
        self.bank_money += self.interest_rate_PD_adjusted * np.sum(self.d)


    def _pay_interest_and_check_bankruptcy(self):
        """Interest is payed first, but the companies that went bankrupt from this do not pay the bank any money.
        Tries to model the idea that bank requires interest payment, but because the company is unable to pay it, they go bankrupt instead.
        """
        self.m -= self.interest_rate_PD_adjusted * self.d
        self._company_bankruptcy_check()
        self.bank_money += self.interest_rate_PD_adjusted * np.sum(self.d)


    def _mutate_beta(self, idx_new_companies: np.ndarray) -> None:
        """Given list of indices, update the beta values of those indices to be the beta value value of the company with the highest production plus a small change.

        Args:
            idx_new_companies (np.ndarray): Indices whose beta values are to be mutated
        """
        
        idx_new_companies = idx_new_companies[0]  # Remove dtype
        N_new_companies = idx_new_companies.size
        if N_new_companies > 0:
            if self.beta_update_method == "production":
                # Get beta value of the company with the highest production
                beta_top = self.beta[np.argmax(self.p)]
                # New companies beta values is then the chosen top company's beta value plus a small change
                # mutations = self.beta_mutation_size * (np.random.randint(low=0, high=2, size=N_new_companies) * 2 - 1)  # Randomly choose between -1 and 1
                mutations = np.random.uniform(low=-self.beta_mutation_size, high=self.beta_mutation_size, size=N_new_companies)
                
                self.beta[idx_new_companies] = beta_top + mutations

            # Second option: Each bankrupt company picks a random beta value from the surviving companies and adds a mutation            
            elif self.beta_update_method == "random":            
                # Get index of surviving companies by removing the bankrupt companies
                idx_surviving_companies = np.arange(self.N)[~np.isin(np.arange(self.N), idx_new_companies)]
                # Choose a random beta value from the surviving companies for each bankrupt company
                beta_from_surviving = np.random.choice(a=self.beta[idx_surviving_companies], size=N_new_companies)
                # Add a mutation to the chosen beta values
                mutations = np.random.uniform(low=-self.beta_mutation_size, high=self.beta_mutation_size, size=N_new_companies)
                self.beta[idx_new_companies] = beta_from_surviving + mutations    
                
            else:
                raise ValueError(f"{self.beta_update_method} is not a valid value for beta_update_method.")
        
        self.beta[idx_new_companies] = np.maximum(self.beta[idx_new_companies], 0)  # Prevent companies from ~never taking loans


    def _company_bankruptcy_check(self) -> None:
        """Bankrupt if p + m < d. If goes bankrupt, first pay any remaining money to the bank, 
        then start a new company in its place with initial values.
        """
        # Find all companies who have gone bankrupt
        bankrupt_idx = np.where(self.p + self.m < self.d * self.interest_rate_PD_adjusted)
        self.N_bankrupt = bankrupt_idx[0].size
        
        # If no banks went bankrupt, do nothing
        if self.N_bankrupt > 0:
            # Companies with positive money pay it to the bank, but no more than the original debt.
            # Elementwise minimum of debt and money of bankrupt companies with positive money.
            # Bank gets the sum of this minimum as money
            bankrupt_companies_with_positive_money_idx = np.where(self.m[bankrupt_idx] > 0)
            money_from_bankrupt_companies = self.m[bankrupt_idx][bankrupt_companies_with_positive_money_idx]
            debt_from_bankrupt_companies = self.d[bankrupt_idx][bankrupt_companies_with_positive_money_idx]
            minimum_of_money_and_debt_from_bankrupt_companies = np.minimum(money_from_bankrupt_companies, debt_from_bankrupt_companies)
            self.bank_money += np.sum(minimum_of_money_and_debt_from_bankrupt_companies)
            
            # Record the fraction of money paid back to the bank compared to the original debt
            fraction_of_money_paid_back = np.sum(minimum_of_money_and_debt_from_bankrupt_companies) / np.sum(debt_from_bankrupt_companies)
            # print(f"%m paid back to the bank = {fraction_of_money_paid_back * 100 :.3f}, N_bankrupt = {self.N_bankrupt}")
            
            # Start new companies, i.e. set bankrupt companies to the initial values
            self.m[bankrupt_idx] = 1.
            self.d[bankrupt_idx] = 0.
            self.p[bankrupt_idx] = 1.
            
            # Mutate beta values of the new companies
            self._mutate_beta(bankrupt_idx)
        

    def _first_deriv_backwards(self, time_step: int) -> float:
        """Compute the first derivative of the bank fortune at the current time using a backwards stencil.
        A backwards stencil is used as we do not have future values.

        Returns:
            float: First derivative of the interest rate at the current time
        """
        F_last_three = self.bank_fortune_hist[time_step - 4 : time_step -1 ]  # Current timestep has not been updated yet
        backwards_stencil = np.array([1/2, -2, 3/2])
        dt = 1  # dt does not matter, as only interested in if positive or negative
        return np.dot(F_last_three, backwards_stencil) / dt


    def _second_deriv_backwards(self, time_step) -> float:
        """Compute the second derivative of the interest rate at the current time using a backwards stencil.

        Returns:
            float: Second derivative of the interest rate at the current time
        """
        F_last_four = self.bank_fortune_hist[time_step - 5 : time_step - 1]  # Current timestep has not been updated yet
        backwards_stencil = np.array([-1, 4, -5, 2])
        dt = 1  # dt does not matter, only sign of derivative does
        return np.dot(F_last_four, backwards_stencil) / dt ** 2


    def _set_interest_rate(self, time_step):
        """Bank sets a new interest rate.
        If the bank has experienced positive growth, it repeats the previus interest rate adjustment, but if the growth was negative, it does the opposite adjustment.
        """
        # Ensure enough data points for derivative. 5 is the minimum number of data points needed for the second derivative.
        if time_step >= 5:  
            # Two possible options for defining growth: first derivative or second derivative. 
            # Use whichever is wanted, determinted by self.derivative_order. Both are stored for plotting.
            self.derivative_1st = self._first_deriv_backwards(time_step)
            self.derivative_2nd = self._second_deriv_backwards(time_step)

            deriv_list = [self.derivative_1st, self.derivative_2nd]
            deriv = deriv_list[self.derivative_order - 1]  # -1 to match 0-indexing and derivative order

            # Change interest action (i.e. swap from decrease to increase or vice versa) if derivative changes sign.            
            if self.interest_action != np.sign(deriv):
                self.interest_action *= -1
            
            # Update interest rate to be a percent of its previous value
            # Positive increasement are adjusted to match the inherent larger decreasements
            if self.interest_action == 1:  
                negative_bias_correction_factor = 1 / (1 - self.interest_rate_change_size)  # Found analytically
                self.interest_rate = self.interest_rate * (1 + negative_bias_correction_factor * self.interest_rate_change_size)
            else:  # self.interest_action == -1
                self.interest_rate = self.interest_rate * (1 - self.interest_rate_change_size)
            

    def _adjust_interest_for_default_probability(self, time_step):
        # Calculate mean default probability of the last 10 time steps and find the probability of default adjusted interest rate
        mean_length = 10
        if time_step >= mean_length:
            prob_default = np.mean(self.N_bankrupt_hist[time_step - mean_length: time_step - 1]) / self.N
            prob_default = np.min((prob_default, 0.99))  # Ensure probability is not above 1
            self.interest_rate_PD_adjusted =  self.interest_rate * prob_default /((1 + self.interest_rate) * (1 - prob_default)) # (1 + self.interest_rate) / (1 - prob_default) - 1
            # self.interest_rate_PD_adjusted = (1 + self.interest_rate) / (1 - prob_default) - 1
        # self.interest_rate_PD_adjusted = self.interest_rate  # OBS only use this if want to exclude the default probability adjustment
        self.interest_rate_PD_adjusted = np.max((self.interest_rate_PD_adjusted, 1e-3))
        

    def _store_values_in_hist_arrays(self, time_step):
        # Store values
        self.p_hist[:, time_step] = self.p * 1
        self.d_hist[:, time_step] = self.d * 1
        self.m_hist[:, time_step] = self.m * 1
        self.beta_hist[:, time_step] = self.beta * 1
        self.bank_fortune_hist[time_step] = self.bank_money + np.sum(self.d)
        self.interest_rate_hist[time_step] = self.interest_rate * 1
        self.interest_rate_PD_adjusted_hist[time_step] = self.interest_rate_PD_adjusted * 1
        self.N_bankrupt_hist[time_step] = self.N_bankrupt * 1
        # Store derivs for plotting
        self.first_derivative_hist[time_step] = self.derivative_1st
        self.second_derivative_hist[time_step] = self.derivative_2nd

        self.did_not_take_loan_hist[time_step] = self.did_not_take_loan * 1
        self.did_not_take_loan = 0  # Reset to 0 for next time step


    def _data_to_file(self) -> None:
        # Collect data to store in one array
        all_data = np.zeros((self.N, self.time_steps, 8))
        all_data[:, :, 0] = self.p_hist
        all_data[:, :, 1] = self.d_hist
        all_data[:, :, 2] = self.m_hist
        all_data[:, :, 3] = self.bank_fortune_hist
        all_data[0, :, 4] = self.interest_rate_hist
        all_data[1, :, 4] = self.interest_rate_PD_adjusted_hist
        all_data[:, :, 5] = self.beta_hist
        all_data[:, :, 6] = self.did_not_take_loan_hist
        all_data[0, :, 7]  = self.first_derivative_hist
        all_data[1, :, 7]  = self.second_derivative_hist

        filename = Path.joinpath(self.dir_path_output, self.file_parameter_addon + ".npy")
        np.save(filename, arr=all_data)


    def simulation(self, func_buyer_seller_idx):
        # Initial values
        self._initialize_market()
        self._initialize_hist_arrays()

        # Time evolution
        for i in tqdm(range(1, self.time_steps)):
            # Allow each company (on average) to be picked once
            for _ in range(self.N):
                # Find seller and buyer idx
                buyer_idx, seller_idx = func_buyer_seller_idx()
                self._transaction(buyer_idx, seller_idx)

            # Pay interest to bank, take loan if in negative, and check for bankruptcies
            self._pay_interest()
            self._negative_money_to_loan()
            self._company_bankruptcy_check()
            # Bank checks its growth and sets interest rate accordingly
            self._set_interest_rate(time_step=i)
            self._adjust_interest_for_default_probability(time_step=i)
            # Store current values in history arrays
            self._store_values_in_hist_arrays(time_step=i)

        # Save the history arrays to a file
        self._data_to_file()


    def repeated_simulation(self, number_of_simulations: int, func_buyer_seller_idx: Callable[[], tuple[int, int]]):
        # Want beta, interest rate and production over multiple simulations
        beta_arr = np.empty((self.N, number_of_simulations))
        interest_rate_arr = np.empty(number_of_simulations)
        interest_rate_PD_adjusted_arr = np.empty(number_of_simulations)
        production_arr = np.empty((self.N, number_of_simulations))
        
        for i in range(number_of_simulations):
            self.simulation(func_buyer_seller_idx)
            beta_arr[:, i] = self.beta
            interest_rate_arr[i] = self.interest_rate
            interest_rate_PD_adjusted_arr[i] = self.interest_rate_PD_adjusted
            production_arr[:, i] = self.p
            
        # Store values
        all_data = np.empty((self.N, number_of_simulations, 4))
        all_data[:, :, 0] = beta_arr
        all_data[:, :, 1] = interest_rate_arr
        all_data[:, :, 2] = interest_rate_PD_adjusted_arr
        all_data[:, :, 3] = production_arr
        
        filename = Path.joinpath(self.dir_path_output, "repeated_" + self.file_parameter_addon + ".npy")
        np.save(filename, arr=all_data)
    

# Parameters
N_companies = 100
time_steps = 5000
alpha = 0.1
interest_rate_change_size = 0.05
beta_mutation_size = 0.1
beta_update_method = "random"  # "production" or "random". Production chooses the beta value of the company with the highest production, random chooses a random beta value from the surviving companies.
derivative_order = 2

if __name__ == "__main__":
    print("You ran the wrong script :\)")
