import numpy as np
from tqdm import tqdm
from pathlib import Path
import h5py


class Workforce():
    def __init__(self, number_of_companies, number_of_workers, salary_increase, time_steps, salary_increase_space=np.linspace(0.01, 0.5, 5), seed=None):
        self.N = number_of_companies
        self.W = number_of_workers 
        self.salary_increase = salary_increase
        self.time_steps = time_steps
        self.salary_increase_space = salary_increase_space  # Used in case the store_peak_rho_space method is not called
        self.seed = seed
        
        # Local paths for saving files
        file_path = Path(__file__)
        self.dir_path = file_path.parent.parent.parent
        self.dir_path_output = Path.joinpath(self.dir_path, "output")
        self.dir_path_image = Path.joinpath(self.dir_path, "images", "image_redistribution_no_m")
        self.dir_path_image.mkdir(parents=True, exist_ok=True)
        self.group_name = f"Steps{self.time_steps}_N{self.N}_W{self.W}_ds{self.salary_increase}"
        
        # Seed
        np.random.seed(self.seed)  
        

    def _initialize_market_variables(self) -> None:    
        # System variables
        self.interest_rate_free = 0.05
        self.interest_rate = self.interest_rate_free
        self.time_scale = 1

        # Company variables
        self.w = np.ones(self.N, dtype=int)  # Will redistribute workers before first timestep anyway
        self.d = np.zeros(self.N, dtype=float)  # Debt
        self.salary = np.random.uniform(self.salary_increase, 1, self.N)  # Pick random salaries
                
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
        self.salary_hist = np.zeros((self.N, self.time_steps)) 
        
        # System
        self.interest_rate_hist = np.zeros(self.time_steps, dtype=float)
        self.went_bankrupt_hist = np.zeros(self.time_steps, dtype=int)
        self.unemployed_hist = np.zeros(self.time_steps, dtype=int)
        self.system_money_spent_hist = np.zeros(self.time_steps, dtype=float)
        
        # Initial values
        self.w_hist[:, 0] = self.w * 1
        self.d_hist[:, 0] = self.d * 1
        
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
        # Include option for not choosing a company to work for i.e. stay unemployed.
        # Have a false company that no one works for, and if a worker chooses this company, they are unemployed.
        prob_get_a_worker = self.salary_including_ghost_company ** 2 / (self.salary_including_ghost_company ** 2).sum()        
        
        # Should this include the probability of bankruptcy?
        # So p_ghost = min(s) + PD  #  or similar
        return prob_get_a_worker


    def _redistribute_workers(self):
        # Include option for not choosing a company to work for i.e. stay unemployed.
        # Use a single "ghost company" to reflect unemployment
        self.w[:] = 0  # "Fire" all workers
        self._ghost_company()
        
        # All workers picks a company to work for with some probability
        idx_each_worker_choose = np.random.choice(np.arange(self.N+1), size=self.W, replace=True, p=self._employment_probability())  # idx of companies that each worker chooses to work for
        idx_company_that_hires_including_ghost_company, number_of_workers_hired_including_ghost_company = np.unique(idx_each_worker_choose, return_counts=True)
        
        # Get rid of ghost company
        idx_company_that_hires, number_of_workers_hired = idx_company_that_hires_including_ghost_company[:-1], number_of_workers_hired_including_ghost_company[:-1]

        # Update values
        self.w[idx_company_that_hires] = number_of_workers_hired
        
        self._employed()


    def _employment_probability_alt(self) -> np.ndarray:
        # OBS USE OTHER FORMULA
        return self.salary ** 2 / (self.salary ** 2).sum()
    

    def _redistribute_workers_alt(self):
        """Every company has a probability to employ a worker. Each worker draws a random company to attempt to work for. If it is not employed, it remains unemployed."""
        # Fire all workers
        self.w[:] = 0
        
        # Workers pick a company to attempt to work for
        idx_workers_chosen_company = np.random.choice(np.arange(self.N), size=self.W, replace=True)  # Includes multiple of the same company
        
        # Get employment probabilities and random numbers to compare these with
        company_employment_probability = self._employment_probability_alt()        
        random_numbers = np.random.uniform(size=self.W)

        # Loop over each company that the workers choose, and see if they are employed
        for i, idx_chosen in enumerate(idx_workers_chosen_company):
            if random_numbers[i] < company_employment_probability[idx_chosen]:
                self.w[idx_chosen] += 1


    def _sell_and_salary(self):
        self._employed()
        # Pick the indices of the companies that will sell
        idx_companies_selling = np.random.choice(np.arange(self.N), size=self.N, replace=True)
        # Find which companies that sells and how many times they do it
        idx_unique, counts = np.unique(idx_companies_selling, return_counts=True)
        # Find how much each of the chosen companies sell for and how much they pay in salary
        sell_unique = self.w[idx_unique] * self.system_money_spent_last_step / self.employed  
        salary_unique = self.salary[idx_unique] * self.w[idx_unique]
        # Multiply that by how many times they sold and paid salaries
        sell_unique_all = sell_unique * counts
        salary_unique_all = salary_unique * counts
        
        # Update values
        self.d[idx_unique] += salary_unique_all - sell_unique_all
        self.system_money_spent += salary_unique_all.sum()
        

    def _pay_interest(self) -> None:   
        positive_debt_idx = self.d > 0
        self.d[positive_debt_idx] *= 1 + self.interest_rate
    

    def _update_salary(self, time_step):
        """Each company changes its salary. 
        If a company expects to loose workers such that w < m, it will increase salary. Otherwise it decreases it.
        """
        # Values for increased and decreased salary
        noise_factor = 1# np.random.uniform(0, 1, size=self.N)
        negative_correction_factor = self.salary_increase * noise_factor / (1 - self.salary_increase * noise_factor)
        
        increased_salary_val = self.salary * (1 + negative_correction_factor)
        decreased_salary_val = self.salary * (1 - self.salary_increase * noise_factor) 
        
        # Find who wants to increase 
        delta_debt = self.d - self.d_hist[:, time_step - 1]
        companies_want_to_increase_salary = delta_debt <= 0        
        
        # Make update
        self.salary = np.where(companies_want_to_increase_salary, increased_salary_val, decreased_salary_val)
        
        # Set minimum salary
        self.salary = np.maximum(self.salary, self.salary_increase)


    def _bankruptcy(self):
        # Goes bankrupt if min(w, m) < rd 
        # bankrupt_idx = self.w < self.interest_rate * self.d
        bankrupt_idx = self.w * self.system_money_spent / self.W < self.interest_rate * self.d
        number_of_companies_gone_bankrupt = bankrupt_idx.sum()  # True = 1, False = 0, so sum gives amount of bankrupt companies

        # System values
        self.went_bankrupt = number_of_companies_gone_bankrupt  
        
        # Company values
        self.w[bankrupt_idx] = 1  # New company one worker
        self.d[bankrupt_idx] = 0  # New company buys 1 machinery and gains debt equal to machinery price

        # Pick salary of non-bankrupt companies and mutate it
        idx_surving_companies = np.arange(self.N)[~bankrupt_idx]
        if idx_surving_companies.size != 0:  # There are non-bankrupt companies            
            new_salary_idx = np.random.choice(idx_surving_companies, size=number_of_companies_gone_bankrupt, replace=True)
            self.salary[bankrupt_idx] = self.salary[new_salary_idx] * 1 # + np.random.uniform(-self.salary_increase, self.salary_increase, size=number_of_companies_gone_bankrupt)
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
        self.salary_hist[:, time_step] = self.salary

        # System variables
        self.interest_rate_hist[time_step] = self.interest_rate * 1
        self._unemployed()
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
        self._redistribute_workers()
        
        # Run simulation
        for i in tqdm(range(1, self.time_steps)):
            self._sell_and_salary()
            self._pay_interest()
            self._update_salary(time_step=i)
            self._redistribute_workers()
            self._bankruptcy()
            self._adjust_interest_for_default(time_step=i)
            self._store_values_in_hist_arrays(time_step=i)         
    

    def store_data(self) -> None:
        # Get data
        self._simulation()
        
        # Check if output directory exists
        self.dir_path_output.mkdir(parents=True, exist_ok=True)
        
        # File name and path
        file_name = "redistribution_no_m_simulation_data.h5"
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
            
            # System variables
            group.create_dataset("interest_rate", data=self.interest_rate_hist)
            group.create_dataset("went_bankrupt", data=self.went_bankrupt_hist)
            group.create_dataset("unemployed", data=self.unemployed_hist)
            group.create_dataset("system_money_spent", data=self.system_money_spent_hist)
            
            # Attributes
            group.attrs["W"] = self.W
            group.attrs["salary_increase"] = self.salary_increase
            group.attrs["salary_increase_space"] = self.salary_increase_space


    def store_data_over_parameter_space(self):
        """Create data for different values of rho and store it in the same file as the simulation data.
        """
        # Fix the seed such that the parameter is the only thing changing
        random_seed = np.random.randint(0, 100000)
        np.random.seed(random_seed)
        
        # Depending on whether rho or machine cost is investigated, change the variable
        N_sim = len(self.salary_increase_space)
        for i, rho in enumerate(self.salary_increase_space):
            print(f"{i+1}/{N_sim}")
            self.salary_increase = rho
            self.group_name = f"Steps{self.time_steps}_N{self.N}_W{self.W}_ds{rho}"
            self.store_data()

            
# Define variables for other files to use
number_of_companies = 250
number_of_workers = 5000
time_steps = 5000
salary_increase = 0.03
salary_increase_space = np.array([0.005, 0.01, 0.03, 0.05])  # At around 0.08 starts to diverge
seed = None

# Other files need some variables
workforce = Workforce(number_of_companies, number_of_workers, salary_increase, time_steps, salary_increase_space, seed)

dir_path_output = workforce.dir_path_output
dir_path_image = workforce.dir_path_image
group_name = workforce.group_name

if __name__ == "__main__":
    workforce.store_data()
    # workforce.store_data_over_parameter_space()
    print("Stored Values")