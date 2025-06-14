import numpy as np
from tqdm import tqdm


class WorkForce():
    def __init__(self, number_of_companies, number_of_workers, salary_increase, interest_rate_free, mutation_magnitude, salary_min, inject_money_time, time_steps, seed=None):
        # Set variables
        self.N = number_of_companies
        self.W = number_of_workers 
        self.ds = salary_increase
        self.rf = interest_rate_free
        self.mutation_magnitude = mutation_magnitude
        self.salary_min = salary_min # Minimum salary allowed
        self.time_steps = time_steps        
        self.inject_money_time = inject_money_time
        self.seed = seed
        
        self.group_name = self._get_group_name()
        self._set_seed(self.seed)    
    
    
    def _set_seed(self, seed=None):
        """Set the seed for the random number generator. 
        """
        # If the instance does not have a seed, set it to None
        self.seed = seed
        np.random.seed(self.seed)
    
            
    def _get_group_name(self):
        if isinstance(self.inject_money_time, list):
            inject_money_time = self.inject_money_time[0]
        else:
            inject_money_time = self.inject_money_time
        return f"Steps{self.time_steps}_N{self.N}_W{self.W}_ds{self.ds}_m{self.mutation_magnitude}_rf{self.rf_name}_alpha{self.prob_exponent}_smin{self.salary_min}_seed{self.seed}_increase{self.who_want_to_increase}_transactionsfactor{self.number_of_transactions_per_step}_injectmoney{inject_money_time}"
    
    
    def _initialize_market_variables(self):
        # Company variables
        self.w = np.ones(self.N, dtype=np.int32)
        self.d = -np.ones(self.N, dtype=np.float32) * self.mutation_magnitude
        self.salary = np.random.uniform(self.salary_min+self.mutation_magnitude, self.salary_min+3*self.mutation_magnitude, self.N)
        
        # Initial values
        if type(self.rf) == str:
            # Pick initial value when rf is a variable
            self.rf = 0.01
        self.r = self.rf
        self.PD = 0
        self.went_bankrupt = 0
        self.went_bankrupt_idx = np.zeros(self.N, dtype=np.bool)
        self.mutations_arr = 0
        self.mu = self.mutation_magnitude * self.W / 2
        self.system_money_spent = self.mu 
        self.T = self._time_scale()
        self.W_not_paid = 0
        self.w_paid = 0
    
    
    def _initialize_history_arrays(self):
        # Company
        self.w_hist = np.zeros((self.N, self.time_steps), dtype=np.int32)
        self.d_hist = np.ones((self.N, self.time_steps), dtype=np.float32) * self.mutation_magnitude
        self.s_hist = np.ones((self.N, self.time_steps), dtype=np.float32)
        # System
        self.r_hist = np.zeros(self.time_steps, dtype=np.float32)
        self.went_bankrupt_hist = np.ones(self.time_steps, dtype=np.int32)
        self.went_bankrupt_idx_hist = np.zeros((self.N, self.time_steps), dtype=np.bool)
        self.mu_hist = np.zeros(self.time_steps, dtype=np.float32)
        self.mutations_hist = np.ones(self.time_steps, dtype=np.float32)
        self.W_not_paid_hist = np.ones(self.time_steps, dtype=np.int32)
        self.w_paid_hist = np.empty(self.time_steps, dtype=np.int32)
        # Initial values of history arrays
        self.w_hist[:, 0] = self.w
        self.d_hist[:, 0] = self.d
        self.s_hist[:, 0] = self.salary
        self.r_hist[0] = self.r
        self.went_bankrupt_hist[0] = self.went_bankrupt
        self.went_bankrupt_idx_hist[:, 0] = self.went_bankrupt_idx
        self.mutations_hist[0] = 0
        self.mu_hist[0] = self.mu
        self.W_not_paid_hist[0] = self.W_not_paid
        self.w_paid_hist[0] = self.w_paid
        
    
    def _store_values_in_history_arrays(self):
        # Company
        self.w_hist[:, self.current_time] = self.w
        self.d_hist[:, self.current_time] = self.d
        self.s_hist[:, self.current_time] = self.salary
        # System
        self.r_hist[self.current_time] = self.r
        self.went_bankrupt_hist[self.current_time] = self.went_bankrupt
        self.went_bankrupt_idx_hist[:, self.current_time] = self.went_bankrupt_idx
        self.mu_hist[self.current_time] = self.mu
        self.mutations_hist[self.current_time] = np.sum(self.mutations_arr)
        self.W_not_paid_hist[self.current_time] = self.W_not_paid
        self.w_paid_hist[self.current_time] = self.w_paid
        
        # Reset values for next step
        self.went_bankrupt = 0
        self.mu = self.system_money_spent
        self.system_money_spent = 0
        

    def _simulation(self):
        # Initialize variables and history arrays
        self._initialize_market_variables()  # Moved to __init__
        self._initialize_history_arrays()
        self._update_workers()  # Run once to get initial values for workers
        
        # Run simulation
        for i in tqdm(range(1, self.time_steps)):
            self.current_time = i
            self._transaction()
            self._inject_money()
            self._pay_interest()
            self._update_salary()
            self._update_workers()
            self._bankruptcy()
            self._adjust_interest_rate()
            self._store_values_in_history_arrays()
            
            