import numpy as np
from tqdm import tqdm
from pathlib import Path


class BankDebtDeflation():
    def __init__(self, number_of_companies: int, money_to_production_efficiency: float, interest_rate_change_size=0.01, beta_mutation_size=0.05, N_interest_update=1, time_steps=1000):
        """Initializer. Children must define the following:
            self.file_parameter_addon

        Args:
            number_of_companies (int): _description_
            money_to_production_efficiency (float): _description_
            time_steps (int): _description_
        """
        self.N = number_of_companies
        self.alpha = money_to_production_efficiency  # Money to production efficiency
        self.interest_rate_change_size = interest_rate_change_size
        self.beta_mutation_size = beta_mutation_size
        self.N_update = N_interest_update  # Number of interest rate updates before a new interest rate is set
        self.time_steps = time_steps  # Steps taken in system evolution.

        # Local paths for saving files.
        file_path = Path(__file__)
        self.dir_path = file_path.parent
        self.dir_path_output = Path.joinpath(self.dir_path, "output")
        self.dir_path_image = Path.joinpath(self.dir_path, "image_bank")
        self.file_parameter_addon_base = f"Steps{self.time_steps}_N{self.N}_alpha{self.alpha}_dr{interest_rate_change_size}_dbeta{beta_mutation_size}"

        # Other parameters are initialized in _initialize_market


    def _initialize_market(self):
        self.p = np.ones(self.N) * 1.  # * 1. converts to float
        self.d = np.zeros(self.N) * 1.
        self.m = np.ones(self.N) * 1.
        self.beta = np.random.uniform(low=0, high=1, size=self.N)  # Risk factor in loans
        self.bank_money = 1#self.N  # Choose another value? N?
        self.interest_rate = 0.1  # OBS might choose another value. 0 could be okay, as system probably needs warmup anyway. Could also mean too safe a start prolonging warmup.
        self.derivative_list = []
        self.interest_policy = 1  # -1 means reduce, 1 means increase
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
        self.did_not_take_loan_hist = np.empty(self.time_steps)
        self.first_derivative_hist = []
        self.second_derivative_hist = []

        # Set initial values
        self.p_hist[:, 0] = self.p * 1
        self.d_hist[:, 0] = self.d * 1
        self.m_hist[:, 0] = self.m * 1
        self.beta_hist[:, 0] = self.beta
        self.bank_fortune_hist[0] = self.bank_money * 1  # No debt initially
        self.interest_rate_hist[0] = self.interest_rate * 1
        self.did_not_take_loan_hist[0] = self.did_not_take_loan * 1  # Must have initial value.


    def _take_loan(self, buyer_idx, seller_idx):
        # Establish loan boundaries and clip the smart loan to match the boundaries
        debt_max = self.p[seller_idx] - self.m[buyer_idx]  # No reason to go beyond the seller's production
        debt_min = 0  # Cannot take negative debt
        delta_debt = (self.beta[buyer_idx] * self.p[buyer_idx] + self.m[buyer_idx]) / self.interest_rate - self.d[buyer_idx]
        delta_debt_clipped = np.clip(delta_debt, a_min=debt_min, a_max=debt_max)

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


    def _pay_interest(self):
        self.m -= self.interest_rate * self.d
        self.bank_money += self.interest_rate * np.sum(self.d)


    def _pay_interest_and_check_bankruptcy(self):
        """Interest is payed first, but the companies that went bankrupt from this do not pay the bank any money.
        Tries to model the idea that bank requires interest payment, but because the company is unable to pay it, they go bankrupt instead.
        """
        self.m -= self.interest_rate * self.d
        self._company_bankruptcy_check()
        self.bank_money += self.interest_rate * np.sum(self.d)


    def _mutate_beta(self, idx_new_companies: np.ndarray):
        """Given list of indices, update the beta values of those indices to be the beta value value of the company with the highest production plus a small change.

        Args:
            idx_new_companies (np.ndarray): Indices whose beta values are to be mutated
        """
        # Choose which of the update methods to use
        pick_best_production = True
        
        # Only run if there actually is any bankrupt companies
        idx_new_companies = idx_new_companies[0]  # Remove dtype
        N_new_companies = idx_new_companies.size
        if N_new_companies > 0:
            if pick_best_production:
                # Get beta value of the company with the highest production
                beta_top = self.beta[np.argmax(self.p)]
                # New companies beta values is then the chosen top company's beta value plus a small change
                mutations = np.random.uniform(low=-self.beta_mutation_size, high=self.beta_mutation_size, size=N_new_companies)
                self.beta[idx_new_companies] = beta_top + mutations

            # Second option: Each bankrupt company picks a random beta value from the surviving companies and adds a mutation            
            else:            
                # Get index of surviving companies by removing the bankrupt companies
                idx_surviving_companies = np.arange(self.N)[~np.isin(np.arange(self.N), idx_new_companies)]
                # Choose a random beta value from the surviving companies for each bankrupt company
                beta_from_surviving = np.random.choice(a=self.beta[idx_surviving_companies], size=N_new_companies)
                # Add a mutation to the chosen beta values
                mutations = np.random.uniform(low=-self.beta_mutation_size, high=self.beta_mutation_size, size=N_new_companies)
                self.beta[idx_new_companies] = beta_from_surviving + mutations    


    def _company_bankruptcy_check(self):
        """Bankrupt if p + m < d. If goes bankrupt, start a new company in its place with initial values.
        Intuition is that when m=0, you must have high enough production to pay off the debt in one transaction i.e. p > d"""
        # Find all companies who have gone bankrupt
        bankrupt_idx = np.where(self.p + self.m < self.d * self.interest_rate)

        # Set their values to the initial values
        self.m[bankrupt_idx] = 1.
        self.d[bankrupt_idx] = 0.
        self.p[bankrupt_idx] = 1.
        # Mutate beta values i.e. give each bankrupt company a living company's beta value and change it slightly
        self._mutate_beta(bankrupt_idx)


    def _first_deriv_backwards(self, time_step: int) -> float:
        """Compute the first derivative of the bank fortune at the current time using a backwards stencil.
        A backwards stencil is used as we do not have future values.

        Returns:
            float: First derivative of the interest rate at the current time
        """
        F_last_three = self.bank_fortune_hist[time_step - 3 : time_step]
        backwards_stencil = np.array([1/2, -2, 3/2])
        dt = 1  # dt does not matter, as only interested in if positive or negative
        return np.dot(F_last_three, backwards_stencil) / dt


    def _second_deriv_backwards(self, time_step) -> float:
        """Compute the second derivative of the interest rate at the current time using a backwards stencil.

        Returns:
            float: Second derivative of the interest rate at the current time
        """
        F_last_four = self.bank_fortune_hist[time_step - 4 : time_step]
        backwards_stencil = np.array([-1, 4, -5, 2])
        dt = 1  # dt does not matter, only sign of derivative does
        return np.dot(F_last_four, backwards_stencil) / dt ** 2


    def _set_interest_rate(self, time_step):
        """Bank sets a new interest rate.
        If the bank has experienced positive growth, it repeats the previus interest rate adjustment, but if the growth was negative, it does the opposite adjustment.
        """
        if time_step >= 4:  # Ensure enough data points for derivative
            # Two possible options for defining growth: first derivative or second derivative
            # Find derivative and store it
            derivative_1st = self._first_deriv_backwards(time_step)
            derivative_2nd = self._second_deriv_backwards(time_step)
            self.derivative_list.append(1 * derivative_2nd)

            # Store derivs for plotting
            self.first_derivative_hist.append(derivative_1st)
            self.second_derivative_hist.append(derivative_2nd)


        if len(self.derivative_list) == self.N_update:
            # Find mean of the last N_update interest rates derivatives
            mean_derivative = np.mean(self.derivative_list)
            # If doing bad, do the opposite action as before i.e. flip sign of interest_policy.
            # If doing good, no change is made
            if mean_derivative <= 1e-5:
                self.interest_policy *= -1

            # Update interest rate
            self.interest_rate = self.interest_rate * (1 + self.interest_policy * self.interest_rate_change_size)
            # Empty list of derivatives
            self.derivative_list = []


    def _data_to_file(self) -> None:
        # Collect data to store in one array
        all_data = np.empty((self.N, self.time_steps, 8))
        all_data[:, :, 0] = self.p_hist
        all_data[:, :, 1] = self.d_hist
        all_data[:, :, 2] = self.m_hist
        all_data[:, :, 3] = self.bank_fortune_hist
        all_data[:, :, 4] = self.interest_rate_hist
        all_data[:, :, 5] = self.beta_hist
        all_data[:, :, 6] = self.did_not_take_loan_hist
        all_data[0, :-4, 7]  = np.array(self.first_derivative_hist)
        all_data[1, :-4, 7]  = np.array(self.second_derivative_hist)

        filename = Path.joinpath(self.dir_path_output, self.file_parameter_addon + ".npy")
        np.save(filename, arr=all_data)


    def _store_values_in_hist_arrays(self, time_step):
        # Store values
        self.p_hist[:, time_step] = self.p * 1
        self.d_hist[:, time_step] = self.d * 1
        self.m_hist[:, time_step] = self.m * 1
        self.beta_hist[:, time_step] = self.beta * 1
        self.bank_fortune_hist[time_step] = self.bank_money + np.sum(self.d)
        self.interest_rate_hist[time_step] = self.interest_rate * 1
        self.did_not_take_loan_hist[time_step] = self.did_not_take_loan * 1
        self.did_not_take_loan = 0  # Reset to 0 for next time step


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

            # Pay interest to bank and check for bankruptcies
            # self._pay_interest()
            # self._company_bankruptcy_check()
            self._pay_interest_and_check_bankruptcy()
            # Bank checks its growth and sets interest rate accordingly
            self._set_interest_rate(time_step=i)
            # Store current values in history arrays
            self._store_values_in_hist_arrays(time_step=i)

        # Save the history arrays to a file
        self._data_to_file()


# Parameters
N_companies = 100
time_steps = 5000
alpha = 0.05
interest_rate_change_size = 0.05
beta_mutation_size = 0.08
N_interest_update = 1

if __name__ == "__main__":
    print("You ran the wrong script :)")
