import numpy as np
from master import WorkForce


class MethodsWorkForce(WorkForce):
    def __init__(self, number_of_companies, number_of_workers, salary_increase, interest_rate_free, mutation_magnitude, prob_exponent, update_methods: dict, time_steps, seed):
        """Must define the following methods for master to work:
            _transaction()
            _pay_interest()
            _update_salary()
            _bankruptcy()
            _update_workers()
            _adjust_interest_rate()
            _time_scale()
        """
        self.worker_update_method = update_methods["worker_update"]
        self.bankruptcy_method = update_methods["bankruptcy"]
        self.mutation_method = update_methods["mutation"]
        
        super().__init__(number_of_companies, number_of_workers, salary_increase, interest_rate_free, mutation_magnitude, time_steps, seed)
        
        # Initial values and choice of methods
        self._pick_functions()
        

    def _pick_functions(self):
        """The methods used are defined.
        """
        self._pick_bankrupt_salary_func()      
        self._pick_worker_update_func()  
        self._pick_interest_update_func()
        self._pick_bankruptcy_func()


    def _pick_worker_update_func(self):
        if self.worker_update_method == "unlimited":
            self.W = self.w.sum()
            self._update_workers = self._update_workers_unlimited
        elif self.worker_update_method == "limited":
            self._update_workers = self._update_workers_limited
    
    
    def _pick_bankrupt_salary_func(self):
        if self.mutation_method == "lastT":
            self.func_bankrupt_salary = self._bankrupt_salary_lastT
        elif self.mutation_method== "spread":
            self.func_bankrupt_salary = self._bankrupt_salary_spread
        elif self.mutation_method == "constant":
            self.func_bankrupt_salary = self._bankrupt_salary_mutate
        elif self.mutation_method == "minimum":
            self.func_bankrupt_salary = self._bankrupt_salary_minimum
        elif self.mutation_method == "log":
            self.func_bankrupt_salary = self.bankrupt_salary_log
        elif self.mutation_method == "mean":
            self.func_bankrupt_salary = self._bankrupt_salary_mean
        else:
            print("Mutation magnitude must be `constant`, 'spread' or 'lastT', 'minimum', ")
    
    
    def _pick_interest_update_func(self):
        # Variable
        if self.rf_name == "variable":
            self._adjust_interest_rate = self._update_rf            
        # float
        elif type(self.rf_name) == float:
            self._adjust_interest_rate = self._adjust_rf_for_PD
        else:
            print("Interest rate must be a float or 'variable'")


    def _pick_bankruptcy_func(self):
        if self.bankruptcy_method == "cannot_pay_salary":
            self._bankruptcy = self._bankruptcy_cannot_pay_salary
        elif self.bankruptcy_method == "negative_money":
            self._bankruptcy = self._bankruptcy_negative_money
        else:
            print("Bankruptcy method must be 'cannot_pay_salary' or 'negative_money'")


    def _transaction(self):
        # Pick the indices of the companies that will sell
        idx_companies_selling = np.random.choice(np.arange(self.N), size=self.N, replace=True)
        # Find which companies that sells and how many times they do it
        idx_unique, counts = np.unique(idx_companies_selling, return_counts=True)
        # Find how much each of the chosen companies sell for and how much they pay in salary
        sell_unique = self.w[idx_unique] * self.mu / self.W  
        salary_unique = self.salary[idx_unique] * self.w[idx_unique]
        # Multiply that by how many times they sold and paid salaries
        sell_unique_all = sell_unique * counts
        salary_unique_all = salary_unique * counts
        # Update values
        self.d[idx_unique] += salary_unique_all - sell_unique_all
        self.system_money_spent += salary_unique_all.sum()
        

    def _pay_interest(self):   
        """Companies with positive debt pay interest rates
        """
        positive_debt_idx = self.d > 0
        money_spent_on_interest = self.d[positive_debt_idx] * self.r
        self.d[positive_debt_idx] += money_spent_on_interest
        
    
    def _update_salary(self):
        """All companies update their salaries depending on whether they made a profit or not. 
        """
        additive_noise = 0 #np.random.uniform(-0.05, 0.05, size=self.N)
        noise_factor = np.random.uniform(0, 1, size=self.N)
        ds_noise = self.ds * noise_factor
        negative_correction = 1 / (1 + np.abs(ds_noise))
        # Values after update
        ds_pos = self.salary * (1 + ds_noise)
        ds_neg = self.salary * (1 - ds_noise * negative_correction)
        # Find who wants to increase i.e. who lowered their debt
        want_to_increase = self.d - self.d_hist[:, self.current_time - 1] <= 0
        # Perform update and enforce minimum salary
        self.salary = np.where(want_to_increase, ds_pos, ds_neg) + additive_noise
        self.salary = np.maximum(self.salary, self.salary_min)
        
    
    def _update_salary_mutate(self):
        """All companies update their salaries depending on whether they made a profit or not.
        The salaries have a "noisy" term added, q ~ U(-mu/W, mu/W). This replaces mutations in bankruptices.
        """
        q = np.random.uniform(-self.mu / self.W, self.mu / self.W, size=self.N)
        # Noise factor and negative correction
        noise_factor = np.random.uniform(0, 1, size=self.N)
        ds_noise = self.ds * noise_factor
        negative_correction = ds_noise / (1 - ds_noise)
        # Values after update
        ds_pos = self.salary * (1 + negative_correction)
        ds_neg = self.salary * (1 - ds_noise)
        # Determine who increases and decreases
        want_to_increase = self.d - self.d_hist[:, self.current_time - 1] <= 0
        # Perform update, add noise and enforce minimum salary
        self.salary = np.where(want_to_increase, ds_pos, ds_neg)
        self.salary += q
        self.salary = np.maximum(self.salary, self.salary_min)
        
        
    def _update_workers_limited(self):
        """All workers are "fired", then rehired by the companies proportionally to the companies' salary. 
        """
        # "Fire" workers
        self.w[:] = 0
        # All workers choose a company to work for proportionally to the salary
        prob_choose = self.salary ** self.prob_exponent / (self.salary ** self.prob_exponent).sum()  
        idx_worker_choose = np.random.choice(np.arange(self.N), size=self.W, replace=True, p=prob_choose)
        idx_company_workers_went_to, number_of_workers_went_to_company = np.unique(idx_worker_choose, return_counts=True)
        # Update number of workers
        self.w[idx_company_workers_went_to] = number_of_workers_went_to_company
        
        
    def _update_workers_unlimited(self):
        """Workers are an unlimited resource. Companies that made money hires more workers, and vice versa. 
        Two options for the amount of workers a company can hire/fire:
            1. w changes by +-1.
            2. w changes depending on economy health. In good times, willing to hire many and fire few, in good times the opposite. 
                This reflects optimism/pessism. For example a company  that had a bad selling in a good economy believes it can sell more in the future and do not want to get rid of many workers.
        """
        who_made_a_profit = self.d - self.d_hist[:, self.current_time - 1] <= 0
        # Option 1
        w_increased = self.w + 1
        w_decreased = self.w - 1
        # Option 2 
        # *Write option 2*
        # Option 3 - Everyone has equal workers
        # self.w[:] = 100
        
        # # Apply update and make sure w is non-negative        
        self.w = np.where(who_made_a_profit, w_increased, w_decreased)
        self.w = np.maximum(self.w, 1)
        self.W = self.w.sum()
    
        
    def _bankruptcy_cannot_pay_salary(self):
        """Companies who pays more in debt than they earn from selling go bankrupt. 
        """
        # Find who goes bankrupt
        bankrupt_idx = self.w * self.mu / self.W < self.r * self.d
        self.went_bankrupt = bankrupt_idx.sum()
        # Reset their values
        self.w[bankrupt_idx] = 0  # OBS Should this be 1?
        self.d[bankrupt_idx] = 0
        self.func_bankrupt_salary(bankrupt_idx)
        self.salary = np.maximum(self.salary, self.salary_min)
        
        if self.worker_update_method == "unlimited":
            self.W = self.w.sum()
    
        if bankrupt_idx[0] == True:
            self.first_company_went_bankrupt = 1
        else:
            self.first_company_went_bankrupt = 0

    
    
    def _bankruptcy_negative_money(self):    
        """A company goes bankrupt if it any debt i.e. no money
        """
        bankrupt_idx = self.d > 0
        self.went_bankrupt = bankrupt_idx.sum()
        # Reset values / create new companies
        self.w[bankrupt_idx] = 0
        self.d[bankrupt_idx] = -1e-8
        self.func_bankrupt_salary(bankrupt_idx)
        self.salary = np.maximum(self.salary, self.salary_min)
        
        if bankrupt_idx[0] == True:
            self.first_company_went_bankrupt = 1
        else:
            self.first_company_went_bankrupt = 0
    
    
    def _bankrupt_salary_mutate(self, bankrupt_idx):
        """Bankrupt companies pick a non-bankrupt company and mutate their salary.

        Args:
            bankrupt_idx (np.ndarray): List of companies who went bankrupt
        """
        # Sample indices from non-bankrupt companies with a positive number of workers
        idx_not_bankrupt = ~bankrupt_idx
        idx_positive_workers = self.w > 0
        idx_not_bankrupt_with_positive_workers = idx_not_bankrupt & idx_positive_workers
        idx_surviving_companies = np.arange(self.N)[idx_not_bankrupt_with_positive_workers]
        self.mutations_arr = np.random.uniform(-self.mutation_magnitude, self.mutation_magnitude, size=self.went_bankrupt)
        if len(idx_surviving_companies) != 0:
            new_salary_idx = np.random.choice(idx_surviving_companies, replace=True, size=self.went_bankrupt)
            self.salary[bankrupt_idx] = self.salary[new_salary_idx] + self.mutations_arr
            self.salary = np.maximum(self.salary, self.salary_min)
        else:
            self.salary = np.random.uniform(self.salary_min, np.max(self.salary[bankrupt_idx]), self.N)
            self.mutations_arr = 0
        
    
    def _bankrupt_salary_lastT(self, bankrupt_idx):
        # Salaries are picked randomly from the last T time steps' salaries.
        # If T is larger than the current time, the salaries are picked from sets up until now.
        start_time = np.maximum(0, self.current_time - self.T)
        set_of_salaries = self.s_hist[:, start_time : self.current_time].flatten()
        self.salary[bankrupt_idx] = np.random.choice(set_of_salaries, size=self.went_bankrupt, replace=True)
        self.salary = np.maximum(self.salary, self.salary_min)
    
    
    def _bankrupt_salary_spread(self, bankrupt_idx):
        """Salaries are picked randomly among the surviving companies' salaries, and the mutation magnitude is based on the spread / mean of the salaries.

        Args:
            bankrupt_idx (_type_): _description_
        """
        idx_surviving_companies = np.arange(self.N)[~bankrupt_idx]
        mutation_magnitude = np.std(self.salary) / np.mean(self.salary)
        if len(idx_surviving_companies) != 0:
            # Mutation magnitude is the percent change from the salary picked. 
            
            new_salary_idx = np.random.choice(idx_surviving_companies, replace=True, size=self.went_bankrupt)
            # sign_of_mutation = np.random.randint(0, 2, size=self.went_bankrupt) * 2 - 1
            # s_increased = self.salary[new_salary_idx] * (1 + mutation_magnitude / (1 - mutation_magnitude))
            # s_decreased = self.salary[new_salary_idx] * (1 - mutation_magnitude)
            # self.salary[bankrupt_idx] = np.where(sign_of_mutation == 1, s_increased, s_decreased)
            self.mutations_arr = np.random.uniform(-mutation_magnitude, mutation_magnitude, size=self.went_bankrupt)
            self.salary[bankrupt_idx] = self.salary[new_salary_idx] + self.mutations_arr
            self.salary = np.maximum(self.salary, self.salary_min)
        else:
            self.salary = np.random.uniform(self.salary_min, np.max(self.salary), self.N)
            self.mutations_arr = 0


    def _bankrupt_salary_minimum(self, bankrupt_idx):
        """Salaries are picked as the minimum living, w>0 company's salary plus a mutation.

        Args:
            bankrupt_idx (_type_): _description_
        """
        # Find the index of the non-bankrupt company with the lowest salary and w>0
        idx_surviving_companies = np.arange(self.N)[~bankrupt_idx]
        idx_living_w_positive = np.logical_and(~bankrupt_idx, self.w > 0)
        min_salary = self.salary[idx_living_w_positive].min()
        
        if len(idx_surviving_companies) != 0:
            self.mutations_arr = np.random.uniform(0, self.mutation_magnitude, size=self.went_bankrupt)
            self.salary[bankrupt_idx] = min_salary + self.mutations_arr
            self.salary = np.maximum(self.salary, self.salary_min)
        else:
            self.salary = np.random.uniform(self.salary_min, np.max(self.salary), self.N)
            self.mutations_arr = 0

    
    def bankrupt_salary_log(self, bankrupt_idx):
        idx_not_bankrupt = ~bankrupt_idx
        idx_positive_workers = self.w > 0
        idx_not_bankrupt_with_positive_workers = idx_not_bankrupt & idx_positive_workers
        idx_surviving_companies = np.arange(self.N)[idx_not_bankrupt_with_positive_workers]
        
        if len(idx_surviving_companies) != 0:
            new_salary_idx = np.random.choice(idx_surviving_companies, replace=True, size=self.went_bankrupt)
            log_salary_chosen = np.log(self.salary[new_salary_idx])
            log_mutation = np.random.uniform(0, self.mutation_magnitude, size=self.went_bankrupt)
            self.salary[bankrupt_idx] = np.exp(log_salary_chosen + log_mutation)
            self.mutations_arr = np.exp(log_mutation)
        else:
            self.salary = np.random.uniform(self.salary_min, np.max(self.salary), self.N)
            self.mutations_arr = 0


    def _bankrupt_salary_mean(self, bankrupt_idx):
        """New companies choose their salary based on the mean salary

        Args:
            bankrupt_idx (_type_): _description_
        """
        idx_not_bankrupt = ~bankrupt_idx
        idx_positive_workers = self.w > 0
        idx_not_bankrupt_with_positive_workers = idx_not_bankrupt & idx_positive_workers
        idx_surviving_companies = np.arange(self.N)[idx_not_bankrupt_with_positive_workers]
        
        if len(idx_surviving_companies) != 0:
            # new_salary_idx = np.random.choice(idx_surviving_companies, replace=True, size=self.went_bankrupt)
            mean_salary = np.mean(self.salary)
            mutation_magnitude = mean_salary * 0.25
            self.mutations_arr = np.random.uniform(-mutation_magnitude, mutation_magnitude, size=self.went_bankrupt)
            self.salary[bankrupt_idx] = mean_salary + self.mutations_arr
            self.salary = np.maximum(self.salary, self.salary_min)
        else:
            self.salary = np.random.uniform(self.salary_min, np.max(self.salary), self.N)
            self.mutations_arr = 0


    def _update_rf(self):
        """The interest rate is updated based on the percent change in mu.
        """
        # The change in the interest rate is a function of the percent change in mu
        epsilon = 5e-2
        r_change = np.log(self.system_money_spent / self.mu) 
        self.rf = self.rf + epsilon * r_change        
        self.rf = np.maximum(self.rf, 1e-5)  # Enforce minimum value for the free interest
        self.r = self.rf
    
    
    def _adjust_rf_for_PD(self):
        self._probability_of_default()
        self.r = (1 + self.rf) / (1 - self.PD) - 1


    def _probability_of_default(self):
        min_time = np.maximum(0, self.current_time - self.T, dtype=np.int32)
        bankrupt_companies_over_time = np.append(self.went_bankrupt_hist[min_time : self.current_time], [self.went_bankrupt])
        PD = np.mean(bankrupt_companies_over_time) / self.N
        self.PD = np.minimum(PD, 0.999)  # Prevent division by zero
        

    def _time_scale(self):
        return 1
        return np.int32(1 / self.r)


    def _store_values_in_history_arrays(self):
        self.T = self._time_scale()
        super()._store_values_in_history_arrays()

    
                        

        
        
        