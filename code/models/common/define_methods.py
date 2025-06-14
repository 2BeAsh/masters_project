import numpy as np
from master import WorkForce


class MethodsWorkForce(WorkForce):
    def __init__(self, number_of_companies, number_of_workers, salary_increase, interest_rate_free, mutation_magnitude, salary_min, number_of_transactions_per_step, update_methods: dict, inject_money_time, time_steps, seed):
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
        self.transaction_method = update_methods["transaction_method"]
        self.include_bankrupt_salary_in_mu = update_methods["include_bankrupt_salary_in_mu"]
        self.who_want_to_increase = update_methods["who_want_to_increase"]
        self.number_of_transactions_per_step = number_of_transactions_per_step
        super().__init__(number_of_companies, number_of_workers, salary_increase, interest_rate_free, mutation_magnitude, salary_min, inject_money_time, time_steps, seed)
        
        # Initial values and choice of methods
        self._pick_functions()
        

    def _pick_functions(self):
        """The methods used are defined.
        """
        self._pick_bankrupt_salary_func()      
        self._pick_worker_update_func()  
        self._pick_interest_update_func()
        self._pick_bankruptcy_func()
        self._pick_want_to_increase()


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
        elif self.mutation_method == "worker_opinion":
            self.func_bankrupt_salary = self._bankrupt_salary_worker_opinion
        elif self.mutation_method == "0_to_mean":
            self.func_bankrupt_salary = self._bankruptcy_salary_0_to_mean
        elif self.mutation_method == "normal":
            self.func_bankrupt_salary = self._bankruptcy_salary_normal
        elif self.mutation_method == "positive_income":
            self.func_bankrupt_salary = self._bankruptcy_salary_positive_income
        elif self.mutation_method == "relative":
            self.func_bankrupt_salary = self._bankrupt_salary_relative
        else:
            print(f"Wrong Mutation magnitude method {self.mutation_method} given, ")
    
    
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


    def _pick_want_to_increase(self):
        """Pick the function that determines which companies update their salary.
        """
        if self.who_want_to_increase == "picked":
            self._want_to_increase = self._only_picked_update_wage
        elif self.who_want_to_increase == "w0":
            self._want_to_increase = self._picked_and_w0_update_wage
        elif self.who_want_to_increase == "all":
            self._want_to_increase = self._non_picked_update_wage
        else: 
            print(f"{self.who_want_to_increase} is not a valid value for who_want_to_increase")


    def _inject_money(self):
        if not isinstance(self.inject_money_time, list):
            self.inject_money_time = [self.inject_money_time]
        if self.current_time in self.inject_money_time:
            self.system_money_spent *= 1.5
    
    
    def _transaction_new(self):
        number_of_companies_transacting = int(self.N * 1.5)
        product_price = self.mu / self.W
        no_legal_companies = 0
        for _ in range(number_of_companies_transacting):
            # Choose company
            idx = np.random.randint(low=0, high=self.N)
            # Perform transaction and update mu
            profit = (self.salary[idx] - product_price) * self.w[idx]
            self.d[idx] += profit
            self.system_money_spent += self.salary[idx] * self.w[idx]
            # Change wage
            negative_correction = 1 / (1 + self.ds)
            if profit > 0:  # lost money
                self.salary[idx] = self.salary[idx] * (1 - self.ds * negative_correction)
            elif profit < 0 or self.w[idx] == 0:
                self.salary[idx] = self.salary[idx] * (1 + self.ds)
            self.salary = np.maximum(self.salary, self.salary_min)            
            
            # Check for bankruptcy
            if self.d[idx] > 0:
                self.went_bankrupt_idx[idx] = True
                self.went_bankrupt += 1
                self.d[idx] = 0
                self.w[idx] = 0
                # Choose random, living company's wage
                nonzero_workers = self.w > 0
                profit = self.d - self.d_hist[:, self.current_time - 1]
                made_profit_nonzero_workers_idx = profit < 0 & nonzero_workers  # debt is negative capital thus <0
                mutation = np.random.uniform(-self.mutation_magnitude, self.mutation_magnitude)
                if made_profit_nonzero_workers_idx.sum() == 0:
                    new_salary = np.random.uniform(self.salary_min, self.salary.max()) + mutation
                    no_legal_companies += 1
                else:
                    idx_new_salary = np.random.choice(np.arange(self.N)[made_profit_nonzero_workers_idx], size=1) 
                    new_salary = self.salary[idx_new_salary] + mutation
                self.salary[idx] = np.maximum(new_salary, self.salary_min)


    def _transaction(self):
        
        if self.transaction_method == "deterministic":
            # All companies are chosen to transact once, but in a random order
            salary_last_time = self.s_hist[:, self.current_time - 1]
            salary_payments = salary_last_time * self.w
            sell_values = self.w * self.mu / self.W            
            transaction_order = np.random.choice(np.arange(self.N), size=self.N, replace=False)
            for idx in transaction_order:        
                # Update
                self.d[idx] += salary_payments[idx] - sell_values[idx]
                
            self.system_money_spent = salary_payments.sum()
        
        else:
            # Use the salaries of last time step
            salary_last_time = self.s_hist[:, self.current_time - 1]
            # Pick the indices of the companies that will sell, Only w>0 companies perform transactions
            number_of_companies_selling = int(np.sum(self.w > 0) * self.number_of_transactions_per_step)
            idx_companies_selling = np.random.choice(np.arange(self.N)[self.w > 0], size=number_of_companies_selling, replace=True)
                            
            # Find which companies that sells and how many times they do it
            idx_unique, counts = np.unique(idx_companies_selling, return_counts=True)
            # Find how much each of the chosen companies sell for
            sell_unique = self.w[idx_unique] * self.mu / self.W / self.number_of_transactions_per_step 
            # Multiply that by how many times they sold
            sell_unique_all = sell_unique * counts
            # Update values
            self.d[idx_unique] -= sell_unique_all
            # Salary depends on "freelance" or not
            
            # Wage i.e. pay workers every time a job is done
            if self.transaction_method == "wage":
                # Do the same as with the capital, find unique and counts.
                salary_unique = salary_last_time[idx_unique] * self.w[idx_unique] / self.number_of_transactions_per_step
                salary_unique_all = salary_unique * counts
                
                if not self.include_bankrupt_salary_in_mu:
                    # If would go in positive debt from paying salaries, set debt to 1e-10 (just have to be >0) and don't pay the salary i.e. don't record it in mu.
                    # Get the indices of companies going bankrupt and not bankrupt
                    idx_goes_bankrupt = np.nonzero(self.d[idx_unique] + salary_unique_all > 0)
                    idx_not_bankrupt = np.nonzero(self.d[idx_unique] + salary_unique_all <= 0)
                    # Set debt of bankrupt to arbitrary small positive number triggering bankrupt criteria.
                    # Update the non-bankrupt companies as usual
                    self.d[idx_goes_bankrupt] = 1e-10  
                    salary_unique_all = salary_unique_all[idx_not_bankrupt]
                    self.d[idx_not_bankrupt] += salary_unique_all
                    
                else:            
                    self.d[idx_unique] += salary_unique_all
                
                self.system_money_spent = salary_unique_all.sum()
                self.w_paid = np.sum(self.w[idx_unique] * counts)
        
            # Salary i.e. pay workers every time step
            elif self.transaction_method == "salary":
                # Salary calculation
                salary_payments = self.w * salary_last_time
                
                if not self.include_bankrupt_salary_in_mu:
                    # If would go in positive debt from paying salaries, set debt to 1e-10 (just have to be >0) and don't pay the salary i.e. don't record it in mu.
                    # Get the indices of companies going bankrupt and not bankrupt (we could perform the update to debt in one line, but want to update salaries as well)
                    idx_goes_bankrupt = np.nonzero(self.d + salary_payments > 0)
                    idx_not_bankrupt = np.nonzero(self.d + salary_payments <= 0)
                    
                    # Update values
                    salary_payments[idx_goes_bankrupt] = 0  # Do not count bankrupt companies in mu
                    self.d += salary_payments
                    self.d[idx_goes_bankrupt] = 1e-10
                    
                else:
                    # All companies pay salaries
                    self.d += salary_payments
                
                self.system_money_spent = salary_payments.sum()    
        

    def _transaction_worker(self):
        """Companies are chosen proportionally to their number of workers, and everyone pays salary regardless of making transactiosn
        """
        # Loop over the number of workers and at each iteration choose a company to sell
        
        # for _ in range(self.W):
        #     # Pick company proportional to number of workers
        #     prob_choose = self.w / self.W
        #     idx_company = np.random.choice(np.arange(self.N), p=prob_choose)
        #     # Update values
        #     self.d[idx_company] -= self.mu / self.W + self.salary[idx_company]
        #     self.system_money_spent += self.salary[idx_company]

        # Alternative method drawing all companies at once and then using unique to update
        idx_company = np.random.choice(np.arange(self.N)[self.w>0], size=self.W, replace=True, p = self.w[self.w>0] / self.w[self.w>0].sum())  # Indices of companies who had an employee chosen to do work
        idx_unique, number_of_workers = np.unique(idx_company, return_counts=True)
        # Find sell and salary values
        sell_unique = number_of_workers * self.mu / self.W
        salary_unique = self.salary[idx_unique] * number_of_workers
        # Update values
        self.d[idx_unique] += salary_unique - sell_unique
        self.system_money_spent += salary_unique.sum()


    def _pay_interest(self):   
        """Companies with positive debt pay interest rates
        """
        positive_debt_idx = self.d > 0
        money_spent_on_interest = self.d[positive_debt_idx] * self.r
        self.d[positive_debt_idx] += money_spent_on_interest
          

    # the usual one:
    def _update_salary(self):
        """All companies update their salaries depending on whether they made a profit or not. 
        """
        negative_correction = 1 / (1 + self.ds)
        # Values after update
        ds_pos = self.salary * (1 + self.ds)
        ds_neg = self.salary * (1 - self.ds * negative_correction)
        # Find who wants to increase i.e. who lowered their debt
        want_to_increase = self._want_to_increase()
        want_to_decrease = self.d - self.d_hist[:, self.current_time - 1] > 0        
        # Perform update and enforce minimum salary
        self.salary[want_to_increase] = ds_pos[want_to_increase]
        self.salary[want_to_decrease] = ds_neg[want_to_decrease]
        self.salary = np.maximum(self.salary, self.salary_min)            
        # self.salary = np.where(want_to_increase, ds_pos, ds_neg)
    
    
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
    
    
    def _bankruptcy_negative_money(self):    
        """A company goes bankrupt if it any debt i.e. no money
        """
        self.went_bankrupt_idx = self.d > 0
        self.went_bankrupt = self.went_bankrupt_idx.sum()
        # Reset values / create new companies
        self.w[self.went_bankrupt_idx] = 0
        self.d[self.went_bankrupt_idx] = 0 #-self.mutation_magnitude
        self.func_bankrupt_salary(self.went_bankrupt_idx)
        self.salary = np.maximum(self.salary, self.salary_min)
        
    
    def _bankrupt_salary_mutate(self, bankrupt_idx):
        """Bankrupt companies pick a non-bankrupt company and mutate their salary.

        Args:
            bankrupt_idx (np.ndarray): List of companies who went bankrupt
        """
        # Sample indices from non-bankrupt companies with a positive number of workers
        N_bankrupt = bankrupt_idx.sum()
        idx_not_bankrupt = ~bankrupt_idx
        idx_positive_workers = self.w > 0
        idx_not_bankrupt_with_positive_workers = idx_not_bankrupt & idx_positive_workers
        idx_surviving_companies = np.arange(self.N)[idx_not_bankrupt_with_positive_workers]
        mutations = np.random.uniform(-self.mutation_magnitude, self.mutation_magnitude, size=N_bankrupt)
        s_before = self.salary * 1
        if len(idx_surviving_companies) != 0:
            new_salary_idx = np.random.choice(idx_surviving_companies, replace=True, size=N_bankrupt)
            salary_pre_max = self.salary[new_salary_idx] + mutations
            self.salary[bankrupt_idx] = np.maximum(salary_pre_max, self.salary_min)
            self.mutations_arr = self.salary - s_before  # Actual mutations taking the max into account
            
        else:
            print("Everyone went bankrupt")
            self.salary = np.random.uniform(self.salary_min, np.max(self.salary), self.N)
            self.mutations_arr = 0

    
    def _bankrupt_salary_worker_opinion(self, bankrupt_idx):
        """When choosing a new salaries, companies pick a random worker and chooses the salary that worker is given. 
        In other words, they choose the salary of a random company proportionally to the number of workers in that company.

        Args:
            bankrupt_idx (_type_): _description_
        """
        # Find indices of companies chosen
        # Since all companies who just went bankrupt have 0 workers, it is unnecessary to exclude bankrupt companies
        P_company = self.w / self.w.sum()  # Since workers have not been redistributed since the bankruptcies, sum(w) != W
        chosen_company_idx = np.random.choice(np.arange(self.N), size=self.went_bankrupt, replace=True, p=P_company)
        
        # Update the salary
        self.salary[bankrupt_idx] = self.salary[chosen_company_idx]

    
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


    def _bankruptcy_salary_0_to_mean(self, bankrupt_idx):
        """New companies draw their salary from a uniform distribution between 0 and the mean salary of the surviving companies.

        Args:
            bankrupt_idx (_type_): _description_
        """
        idx_not_bankrupt = ~bankrupt_idx
        idx_positive_workers = self.w > 0
        idx_not_bankrupt_with_positive_workers = idx_not_bankrupt & idx_positive_workers
        idx_surviving_companies = np.arange(self.N)[idx_not_bankrupt_with_positive_workers]
        
        if len(idx_surviving_companies) != 0:
            mean_salary = np.mean(self.salary[idx_surviving_companies])
            self.salary[bankrupt_idx] = np.random.uniform(0, mean_salary, size=self.went_bankrupt)
            # self.salary = np.maximum(self.salary, self.salary_min)
        else:
            self.salary = np.random.uniform(self.salary_min, np.max(self.salary), self.N)
            self.mutations_arr = 0


    def _bankruptcy_salary_normal(self, bankrupt_idx):
        """New companies darw their salary from a normal distribution with mean and std of the surviving companies' salaries.

        Args:
            bankrupt_idx (_type_): _description_
        """
        idx_not_bankrupt = ~bankrupt_idx
        idx_positive_workers = self.w > 0
        idx_not_bankrupt_with_positive_workers = idx_not_bankrupt & idx_positive_workers
        idx_surviving_companies = np.arange(self.N)[idx_not_bankrupt_with_positive_workers]
        
        if len(idx_surviving_companies) != 0:
            new_salaries = np.random.choice(np.arange(self.N)[idx_surviving_companies], size=self.went_bankrupt, replace=True)
            salary_living = self.salary[idx_surviving_companies]
            mean_salary = np.mean(salary_living)
            std_salary = np.std(salary_living)
            mutations = np.random.normal(loc=0, scale=std_salary, size=self.went_bankrupt)
            self.salary[bankrupt_idx] = new_salaries + mutations
            self.salary = np.maximum(self.salary, self.salary_min)
        else:
            self.salary = np.random.uniform(self.salary_min, np.max(self.salary), self.N)
            self.mutations_arr = 0
            
            
    def _bankruptcy_salary_positive_income(self, bankrupt_idx):
        """New companies get their salary from comapnies who made a profit. If no companies made a profit, they draw from the top 50% of companies with w>0 who lost the least.

        Args:
            bankrupt_idx (_type_): _description_
        """
        # Find the companies that made a profit and did not go bankrupt this time step
        idx_not_bankrupt = ~bankrupt_idx
        positive_workers = self.w > 0
        legal_company = idx_not_bankrupt & positive_workers
        profit = self.d - self.d_hist[:, self.current_time - 1]
        idx_made_a_profit = profit < 0 & legal_company
        N_made_a_profit = idx_made_a_profit.sum()
                
        # Draw from the companies that made a profit
        if N_made_a_profit > 0:
            idx_new_salary = np.random.choice(np.arange(self.N)[idx_made_a_profit], size=self.went_bankrupt, replace=True)
        
        # If no companies made a profit, draw from the top 50% who lost the least
        else:
            idx_sorted = np.argsort(profit)
            idx_top_50p = idx_sorted[:int(self.N / 2)]  # Goes from low to high, and want the profit to be a large negative number
            idx_new_salary = np.random.choice(idx_top_50p, size=self.went_bankrupt, replace=True)

        mutations = np.random.uniform(-self.mutation_magnitude, self.mutation_magnitude, size=self.went_bankrupt)
        self.salary[bankrupt_idx] = self.salary[idx_new_salary] + mutations
            

    def _bankrupt_salary_relative(self, bankrupt_idx):
        """Bankrupt companies pick a non-bankrupt company and relative mutate their salary .

        Args:
            bankrupt_idx (np.ndarray): List of companies who went bankrupt
        """
        # Find the companies that made a profit and did not go bankrupt this time step
        idx_not_bankrupt = ~bankrupt_idx
        positive_workers = self.w > 0
        legal_company = idx_not_bankrupt & positive_workers
        profit = self.d - self.d_hist[:, self.current_time - 1]
        idx_made_a_profit = profit < 0 & legal_company
        N_made_a_profit = idx_made_a_profit.sum()
                
        # Draw from the companies that made a profit
        if N_made_a_profit > 0:
            idx_new_salary = np.random.choice(np.arange(self.N)[idx_made_a_profit], size=self.went_bankrupt, replace=True)
        
        # If no companies made a profit, draw from the top 25% who lost the least
        else:
            idx_sorted = np.argsort(profit)
            idx_top_50p = idx_sorted[:int(self.N / 2)]  # Goes from low to high, and want the profit to be a large negative number
            idx_new_salary = np.random.choice(idx_top_50p, size=self.went_bankrupt, replace=True)

        mutations = np.random.uniform(-self.mutation_magnitude / (1 + np.abs(self.mutation_magnitude)), self.mutation_magnitude, size=self.went_bankrupt)
        self.salary[bankrupt_idx] = self.salary[idx_new_salary] * (1 + mutations)
        
        
    def _only_picked_update_wage(self):
        """ Only companies that were picked to transact updates their salary
        Returns:
            _type_: _description_
        """
        
        return self.d - self.d_hist[:, self.current_time - 1] < 0
    
    
    def _picked_and_w0_update_wage(self):
        """Companies that were picked to transact or companies with w=0 updates their salary
        """
        transact = self.d - self.d_hist[:, self.current_time - 1] < 0
        w0 = self.w == 0
        both = np.logical_or(transact, w0)  # Either transacted or you have zero workers.
        return both
    
    
    def _non_picked_update_wage(self):
        """Companies not picked to transact also update wages        
        """
        return self.d - self.d_hist[:, self.current_time - 1] <= 0


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
        if self.rf != 0.:
            return np.int32(1 / self.r)
        return 1


    def _store_values_in_history_arrays(self):
        self.T = self._time_scale()
        super()._store_values_in_history_arrays()
