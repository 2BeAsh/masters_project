import numpy as np
from tqdm import tqdm
from pathlib import Path
import h5py
from scipy.signal import find_peaks


class Workforce():
    def __init__(self, number_of_companies, number_of_workers, salary_increase, machine_cost, time_steps, salary_increase_space=np.linspace(0.01, 0.5, 5)):
        self.N = number_of_companies
        self.W = number_of_workers 
        self.salary_increase = salary_increase
        self.time_steps = time_steps
        self.machine_cost = machine_cost
        self.salary_increase_space = salary_increase_space  # Used in case the store_peak_rho_space method is not called
        
        # Local paths for saving files
        file_path = Path(__file__)
        self.dir_path = file_path.parent.parent.parent
        self.dir_path_output = Path.joinpath(self.dir_path, "output")
        self.dir_path_image = Path.joinpath(self.dir_path, "images", "image_redistribution")
        self.dir_path_image.mkdir(parents=True, exist_ok=True)
        self.group_name = f"Steps{self.time_steps}_N{self.N}_W{self.W}_rho{self.salary_increase}_M{machine_cost}"
        
        # Seed
        np.random.seed(42)   
        

    def _initialize_market_variables(self) -> None:    
        # System variables
        self.interest_rate_free = 0.05
        self.interest_rate = self.interest_rate_free
        self.time_scale = 12

        # Company variables
        self.w = np.ones(self.N)  # Will redistribute workers before first timestep anyway
        self.d = self.machine_cost * np.ones(self.N, dtype=float)  # Debt
        self.m = np.ones(self.N, dtype=int)  # Machines
        self.salary = np.random.uniform(self.salary_increase, 1+self.salary_increase, self.N)  # Pick random salaries
                
        # Worker variables
        self.unemployed = self.W - self.w.sum()  # Every company starts with employees       

        # Ghost company - for worker redistribution
        self.salary_including_ghost_company = np.append(self.salary, np.min(self.salary))  # Include ghost company that corresponds to unemployed people
        self.w_including_ghost_company = np.zeros(self.N + 1, dtype=int)  # Include ghost company that corresponds to unemployed people

        # Initial values
        self.PD = 0
        self.went_bankrupt = 0
        self.system_money_spent = 0
        self.system_money_spent_last_step = self.w @ self.salary
        
        # Other
        self.negative_correction_factor = 1 / (1 - self.salary_increase)


    def _initialize_hist_arrays(self) -> None:
        # Initialize hist arrays
        # Company
        self.w_hist = np.ones((self.N, self.time_steps))
        self.d_hist = np.zeros((self.N, self.time_steps))
        self.m_hist = np.zeros((self.N, self.time_steps))
        self.salary_hist = np.zeros((self.N, self.time_steps)) 
        
        # System
        self.interest_rate_hist = np.zeros(self.time_steps, dtype=float)
        self.went_bankrupt_hist = np.zeros(self.time_steps, dtype=int)
        self.unemployed_hist = np.zeros(self.time_steps, dtype=int)
        self.system_money_spent_hist = np.zeros(self.time_steps, dtype=float)
        
        # Initial values
        self.w_hist[:, 0] = self.w * 1
        self.d_hist[:, 0] = self.d * 1
        self.m_hist[:, 0] = self.m * 1
        
        self.salary_hist[:, 0] = self.salary * 1
        self.interest_rate_hist[0] = self.interest_rate * 1
        self.went_bankrupt_hist[0] = self.went_bankrupt * 1
        self.unemployed_hist[0] = self.unemployed * 1
        self.system_money_spent_hist[0] = self.system_money_spent * 1

    
    def _unemployed(self): 
        self.unemployed = self.W - self.w.sum()


    def _employed(self):
        self._unemployed()
        self.employed = self.W - self.unemployed
                

    def _production_capacity(self):
        self.p = np.minimum(self.m, self.w)


    def _buy_machinery(self):
        # Companies with less machines than workers buys a machine
        wants_to_buy_machine = self.m < self.w
        # Update values
        self.d[wants_to_buy_machine] += self.machine_cost
        self.m[wants_to_buy_machine] += 1
        
        
    def _ghost_company(self):
        # Fire all workers
        self.w_including_ghost_company[:] = 0

        # Add the minimum salary as the salary for unemployment. People have an equal chance of choosing the worst salary as to stay unemployed
        self.salary_including_ghost_company[:-1] = self.salary
        self.salary_including_ghost_company[-1] = np.min(self.salary)
        
    
    def _employment_probability(self) -> np.ndarray:
        """Probability for each company to get a worker. An additional "ghost company" is added to reflect unemployment. 

        Returns:
            np.ndarray:: self.N + 1 sized array with probabilities for each company to get a worker.
        """
        # Probability of getting employed by a company
        # prob_get_a_worker = self.salary / self.salary.sum()
        # prob_get_a_worker = self.salary ** 2 / (self.salary ** 2).sum()
        
        # Include option for not choosing a company to work for i.e. stay unemployed.
        # Have a false company that no one works for, and if a worker chooses this company, they are unemployed.
        prob_get_a_worker = self.salary_including_ghost_company ** 2 / (self.salary_including_ghost_company ** 2).sum()        
        
        return prob_get_a_worker


    def _redistribute_workers_matrix(self):
        """Each worker draws a random number and compares that to the array of employment probabilities. 
        If the random number is larger than all worker's employment probabilities, the worker remains unemployed.
        If not, it chooses a company proportional to the employment probabilities.
        """
        
        # Fire all workers
        self.w[:] = 0
        
        # Get employment probabilities
        employment_probabilities = self._employment_probability()
        
        # Each workers draws a random number.
        random_numbers = np.random.uniform(0, 1, self.W)  # q
        employment_matrix = random_numbers[:, None] < employment_probabilities[None, :]  # p
        
        # For each worker, find a company that satisfies q < p
        for employ_prob_row in employment_matrix:
            # Calculate the probability for the i'th worker to work for each company
            # If no company is chosen, the worker remains unemployed
            if employ_prob_row.any():
                # Choose a random company amongst those who satisfy q < p to work for
                idx_company = np.random.choice(np.arange(self.N)[employ_prob_row])
                self.w[idx_company] += 1
                

    def _redistribute_workers(self, time_step):
        # Include option for not choosing a company to work for i.e. stay unemployed.
        # Option 1: Have a false company that no one works for, and if a worker chooses this company, they are unemployed.
        # Option 2: Create an array of probabilities for each company, and if a worker does not choose any of those, it stays unemployed.

        # Use a single "ghost company" to reflect unemployment
        self.w[:] = 0
        self._ghost_company()
        
        # All workers picks a company to work for with some probability
        idx_each_worker_choose = np.random.choice(np.arange(self.N+1), size=self.W, replace=True, p=self._employment_probability())  # idx of companies that each worker chooses to work for
        idx_company_that_hires_including_ghost_company, number_of_workers_hired_including_ghost_company = np.unique(idx_each_worker_choose, return_counts=True)
        
        # Get rid of ghost company
        idx_company_that_hires, number_of_workers_hired = idx_company_that_hires_including_ghost_company[:-1], number_of_workers_hired_including_ghost_company[:-1]

        # Cannot hire more than double the number of workers the company had before 
        # If was at 0 workers before, can only hire 1 worker
        # number_of_workers_hired_limited = np.minimum(number_of_workers_hired, 2 * self.w_hist[:, time_step-1][idx_company_that_hires])  # Cannot hire more than twice your last number of workers
        # idx_zero_workers = np.where(self.w_hist[:, time_step-1][idx_company_that_hires] == 0)[0]
        # number_of_workers_hired_limited = number_of_workers_hired.copy()
        # number_of_workers_hired_limited[idx_zero_workers] = np.minimum(number_of_workers_hired[idx_zero_workers], 1)  # If was at 0 workers before, can only hire 1 worker
        
        # Update values
        # self.w[idx_company_that_hires] = number_of_workers_hired_limited
        self.w[idx_company_that_hires] = number_of_workers_hired
        
        self._employed()
        # if self.employed / self.W != 1.0:
        #     print("Employed fraction = ", self.employed / self.W)


    def _sell_and_salary(self):
        """Choose N random companies to sell. Whenever you sell, you also pay salaries.
        """
        # Choose random company to gain money from selling to employed people, but also to pay salaries
        self._production_capacity()
        self._employed()
        for _ in range(self.N):
            idx = np.random.randint(0, self.N)
            sell = self.p[idx] * self.system_money_spent_last_step / self.employed
            salary_paid = self.salary[idx] * self.w[idx]
            
            # Update values
            self.d[idx] = salary_paid - sell
            self.system_money_spent += salary_paid 


    def _pay_interest(self) -> None:   
        positive_debt_idx = self.d > 0
        self.d[positive_debt_idx] *= 1 + self.interest_rate
    

    def _update_salary(self, time_step):
        """Each company changes its salary. 
        If a company expects to loose workers such that w < m, it will increase salary. Otherwise it decreases it.
        """
        # Values for increased and decreased salary
        increased_salary_val = self.salary * (1 + self.salary_increase * self.negative_correction_factor)
        decreased_salary_val = self.salary * (1 - self.salary_increase)
        
        # increased_salary_val = self.salary + self.salary_increase
        # decreased_salary_val = self.salary - self.salary_increase
        
        # Find who wants to increase 
        delta_debt = self.d - self.d_hist[:, time_step - 1]
        companies_want_to_increase_salary = delta_debt <= 0        
        
        # Make update
        # self.salary = np.where(companies_want_to_increase_salary, self.salary + self.salary_increase, self.salary - self.salary_increase)
        self.salary = np.where(companies_want_to_increase_salary, increased_salary_val, decreased_salary_val)


    def _bankruptcy(self):
        # Goes bankrupt if min(w, m) < rd 
        self._production_capacity()
        bankrupt_idx = self.p < self.interest_rate * self.d
        number_of_companies_gone_bankrupt = bankrupt_idx.sum()  # True = 1, False = 0, so sum gives amount of bankrupt companies

        # System values
        self.went_bankrupt = number_of_companies_gone_bankrupt  
        
        # Company values
        self.w[bankrupt_idx] = 1  # New company one worker
        self.d[bankrupt_idx] = self.machine_cost  # New company buys 1 machinery and gains debt equal to machinery price
        self.m[bankrupt_idx] = 1  # New company has 1 machinery

        # Pick salary of non-bankrupt companies and mutate it
        idx_surving_companies = np.arange(self.N)[~bankrupt_idx]
        if idx_surving_companies.size != 0:  # There are non-bankrupt companies            
            new_salary_idx = np.random.choice(idx_surving_companies, size=number_of_companies_gone_bankrupt, replace=True)
            self.salary[bankrupt_idx] = self.salary[new_salary_idx] + (2 * np.random.randint(0, 2, size=number_of_companies_gone_bankrupt) - 1) * self.salary_increase
        else:
            self.salary = np.random.uniform(self.salary_increase, 1, number_of_companies_gone_bankrupt)
        
        # Set minimum salary
        self.salary = np.maximum(self.salary, self.salary_increase)
        
       
    def _probability_of_default(self, time_step, T) -> None:
        if time_step > T:
            bankrupt_T_ago_to_now = np.append(self.went_bankrupt_hist[time_step - T : time_step - 1], [self.went_bankrupt])
            self.PD = np.mean(bankrupt_T_ago_to_now) / self.N
            self.PD = np.minimum(self.PD, 0.99)  # Prevent division by zero.
        
    
    def _adjust_interest_for_default(self, time_step) -> None:
        # Using the probability of default (synonymous with bankruptcy) to adjust the interest rate
        self._probability_of_default(time_step, T=self.time_scale)
        self.interest_rate = (1 + self.interest_rate_free) / (1 - self.PD) - 1 
        
    
    def _store_values_in_hist_arrays(self, time_step: int) -> None:
        # Company variables
        self.w_hist[:, time_step] = self.w
        self.d_hist[:, time_step] = self.d
        self.m_hist[:, time_step] = self.m
        self.salary_hist[:, time_step] = self.salary

        # System variables
        self.interest_rate_hist[time_step] = self.interest_rate * 1
        self._unemployed()
        # print("Number of unemployed: ", self.unemployed, "fraction of total: ", self.unemployed / self.W)
        # print("")
        self.unemployed_hist[time_step] = self.unemployed * 1
        self.went_bankrupt_hist[time_step] = self.went_bankrupt * 1
        self.system_money_spent_hist[time_step] = self.system_money_spent * 1
        
        # Reset values
        self.went_bankrupt = 0  # Reset for next time step
        self.system_money_spent_last_step = self.system_money_spent * 1
        self.system_money_spent = 0  # Reset for next time step
        
        
    def _simulation(self):
        # Initialize variables and hist arrays
        self._initialize_market_variables()
        self._initialize_hist_arrays()
        
        # Redistribute workers once to get initial values
        self._redistribute_workers(time_step=1)
        
        # Run simulation
        for i in tqdm(range(1, self.time_steps)):
            self._buy_machinery()
            self._sell_and_salary()
            self._pay_interest()
            self._update_salary(time_step=i)
            self._redistribute_workers(time_step=i)
            self._bankruptcy()
            self._adjust_interest_for_default(time_step=i)
            self._store_values_in_hist_arrays(time_step=i)         
    
    
    def _peaks(self):
        # Find the amplitude and frequency of system_money_spent peaks using scipy find_peaks
        # Remove the first initial_values_skipped data points as they are warmup
        # Prominence: Take a peak and draw a horizontal line to the highest point between the peak and the next peak. The prominence is the height of the peak's summit above this horizontal line.
        initial_values_skipped = 50
        system_money = self.system_money_spent_hist[initial_values_skipped:]
        prominence = system_money.max() / 4  # Prominence is 1/4 of the max value
        self.peak_idx, _ = find_peaks(x=system_money, height=1, distance=20, width=10, prominence=prominence)  # Height: Minimum y value, distance: Minimum x distance between peaks, prominence: 
        self.peak_vals = system_money[self.peak_idx]
        self.peak_idx += initial_values_skipped 
        
    
    def _get_data(self):
        # Run simulation, peaks
        self._simulation()
        self._peaks()
        

    def store_values(self) -> None:
        # Get data
        self._get_data()
        
        # Check if output directory exists
        self.dir_path_output.mkdir(parents=True, exist_ok=True)
        
        # File name and path
        file_name = "redistribution_simulation_data.h5"
        file_path = self.dir_path_output / file_name
        
        # If the exact filename already exists, open in write, otherwise in append mode
        with h5py.File(file_path, "a") as f:
            # If the group already exists, delete it
            if self.group_name in f:
                del f[self.group_name]

            group = f.create_group(self.group_name)
            
            # Store data in group
            # Company variables
            group.create_dataset("w", data=self.w_hist)
            group.create_dataset("d", data=self.d_hist)
            group.create_dataset("s", data=self.salary_hist)
            group.create_dataset("m", data=self.m_hist)
            
            # System variables
            group.create_dataset("interest_rate", data=self.interest_rate_hist)
            group.create_dataset("went_bankrupt", data=self.went_bankrupt_hist)
            group.create_dataset("unemployed", data=self.unemployed_hist)
            group.create_dataset("system_money_spent", data=self.system_money_spent_hist)
            group.create_dataset("peak_idx", data=self.peak_idx)
            group.create_dataset("peak_vals", data=self.peak_vals)
            
            # Attributes
            group.attrs["W"] = self.W
            group.attrs["salary_increase"] = self.salary_increase
            group.attrs["salary_increase_space"] = self.salary_increase_space


    def store_peak_rho_space(self, seed=42):
        """Get peak values for different salary increase values
        """
        # Fix the seed such that the parameter is the only thing changing
        np.random.seed(seed)
        
        N_sim = len(self.salary_increase_space)
        for i, rho in enumerate(self.salary_increase_space):
            print(f"{i+1}/{N_sim}")
            self.salary_increase = rho
            self.group_name = f"Steps{self.time_steps}_N{self.N}_W{self.W}_rho{rho}_M{self.machine_cost}"
            self.store_values()

            
# Define variables for other files to use
number_of_companies = 250
number_of_workers = 2500
time_steps = 5000
machine_cost = 1
salary_increase = 0.1
salary_increase_space = np.arange(0.04, 0.3, 0.03).round(4)

# Other files need some variables
workforce = Workforce(number_of_companies, number_of_workers, salary_increase, machine_cost, time_steps, salary_increase_space)

dir_path_output = workforce.dir_path_output
dir_path_image = workforce.dir_path_image
group_name = workforce.group_name

if __name__ == "__main__":
    workforce.store_values()
    workforce.store_peak_rho_space()
    print("Stored Values")