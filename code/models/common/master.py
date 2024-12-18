import numpy as np
from tqdm import tqdm


class WorkForce():
    def __init__(self, number_of_companies, number_of_workers, salary_increase, interest_rate_free, mutation_magnitude, time_steps, seed=None):
        # Set variables
        self.N = number_of_companies
        self.W = number_of_workers 
        self.ds = salary_increase
        self.rf = interest_rate_free
        self.time_steps = time_steps
        self.mutation_magnitude = mutation_magnitude
        self.seed = seed
        
        # Parameters
        self.salary_min = 1e-8  # Minimum salary allowed
        self.group_name = self._get_group_name()
        # Set seed
        np.random.seed(seed)
        
        self._initialize_market_variables()
                
        
    def _get_group_name(self):
        return f"Steps{self.time_steps}_N{self.N}_W{self.W}_ds{self.ds}_m{self.mutation_magnitude}_wupdate{self.worker_update_method}_rf{self.rf_name}"
    
    
    def _initialize_market_variables(self):
        # Company variables
        self.w = np.ones(self.N, dtype=np.int32)
        self.d = np.zeros(self.N, dtype=np.float32)
        self.salary = np.random.uniform(self.salary_min, 0.5, self.N)
        
        # Initial values
        if type(self.rf) == str:
            # Pick initial value when rf is a variable
            self.rf = 0.01
        self.r = self.rf
        self.PD = 0
        self.went_bankrupt = 0
        self.mutations_arr = 0
        self.mu = np.mean(self.salary) * self.W
        self.system_money_spent = self.mu
        self.T = self._time_scale()
    
    
    def _initialize_history_arrays(self):
        # Company
        self.w_hist = np.zeros((self.N, self.time_steps), dtype=np.int32)
        self.d_hist = np.zeros((self.N, self.time_steps), dtype=np.float32)
        self.s_hist = np.ones((self.N, self.time_steps), dtype=np.float32)
        # System
        self.r_hist = np.zeros(self.time_steps, dtype=np.float32)
        self.went_bankrupt_hist = np.ones(self.time_steps, dtype=np.int32)
        self.mu_hist = np.zeros(self.time_steps, dtype=np.float32)
        self.mutations_hist = np.ones(self.time_steps, dtype=np.float32)
        # Initial values of history arrays
        self.w_hist[:, 0] = self.w
        self.d_hist[:, 0] = self.d
        self.s_hist[:, 0] = self.salary
        self.r_hist[0] = self.r
        self.went_bankrupt_hist[0] = self.went_bankrupt
        self.mutations_hist[0] = self.mutations_arr
        self.mu_hist[0] = self.mu
        
    
    def _store_values_in_history_arrays(self):
        # Company
        self.w_hist[:, self.current_time] = self.w
        self.d_hist[:, self.current_time] = self.d
        self.s_hist[:, self.current_time] = self.salary
        # System
        self.r_hist[self.current_time] = self.r
        self.went_bankrupt_hist[self.current_time] = self.went_bankrupt
        self.mu_hist[self.current_time] = self.mu
        self.mutations_hist[self.current_time] = np.sum(self.mutations_arr)
        # Reset values for next step
        self.went_bankrupt = 0
        self.mu = self.system_money_spent
        self.system_money_spent = 0
        

    def _simulation(self):
        # Initialize variables and history arrays
        # self._initialize_market_variables()
        self._initialize_history_arrays()
        
        # Run simulation
        for i in tqdm(range(1, self.time_steps)):
            self.current_time = i
            self._transaction()
            self._pay_interest()
            self._update_salary()
            self._update_workers()
            self._bankruptcy()
            self._adjust_interest_rate()
            self._store_values_in_history_arrays()
            
            