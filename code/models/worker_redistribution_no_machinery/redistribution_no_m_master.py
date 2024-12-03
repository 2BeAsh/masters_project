import numpy as np
from tqdm import tqdm
from pathlib import Path
import h5py


class Workforce():
    def __init__(self, number_of_companies, number_of_workers, salary_increase, interest_rate_free, time_steps, ds_space, rf_space, seed=None):
        # Set variables
        self.N = number_of_companies
        self.W = number_of_workers 
        self.salary_increase = salary_increase
        self.interest_rate_free = interest_rate_free
        self.time_steps = time_steps
        self.ds_space = ds_space  # Used in case the store_peak_rho_space method is not called
        self.rf_space = rf_space
        self.seed = seed
        
        # Time scale function
        self._func_time_scale = self._time_scale_0  # Default time scale function
        # self._func_time_scale = self._time_scale_x
        # self._func_time_scale = self._time_scale_inverse_r 
        
        # Local paths for saving files
        file_path = Path(__file__)
        self.dir_path = file_path.parent.parent.parent
        self.dir_path_output = Path.joinpath(self.dir_path, "output")
        self.dir_path_image = Path.joinpath(self.dir_path, "images", "image_redistribution_no_m")
        self.dir_path_image.mkdir(parents=True, exist_ok=True)
        self.salary_min = 1e-8  # Minimum salary allowed
        self.group_name = self._get_group_name()
        
        # File name and path
        file_name = "redistribution_no_m_simulation_data.h5"
        self.file_path = self.dir_path_output / file_name
        
        # Seed
        np.random.seed(self.seed)  
        

    def _get_group_name(self) -> str:
        func_name = self._func_time_scale.__name__.replace("_", "")
        return f"Steps{self.time_steps}_N{self.N}_W{self.W}_ds{self.salary_increase:.3f}_rf{self.interest_rate_free:.3f}_{func_name}"


    def _initialize_market_variables(self) -> None:    
        # Miscellaneous
        
        # System variables
        self.interest_rate = self.interest_rate_free


        # Company variables
        self.w = np.ones(self.N, dtype=int)  # Will redistribute workers before first timestep anyway
        self.d = np.zeros(self.N, dtype=float)  # Debt
        self.salary = np.random.uniform(0.5e-2, 3e-1, self.N)  # Pick random salaries
                
        # Initial values
        self.PD = 0
        self.went_bankrupt = 0
        self.system_money_spent = 0
        self.system_money_spent_last_step = self.w @ self.salary


    def _initialize_hist_arrays(self) -> None:
        # Initialize hist arrays
        # Company
        self.w_hist = np.ones((self.N, self.time_steps))
        self.d_hist = np.zeros((self.N, self.time_steps))
        self.salary_hist = np.zeros((self.N, self.time_steps)) 
        
        # System
        self.interest_rate_hist = np.zeros(self.time_steps, dtype=float)
        self.went_bankrupt_hist = np.zeros(self.time_steps, dtype=int)
        self.system_money_spent_hist = np.zeros(self.time_steps, dtype=float)
        
        # Initial values
        self.w_hist[:, 0] = self.w
        self.d_hist[:, 0] = self.d
        
        self.salary_hist[:, 0] = self.salary
        self.interest_rate_hist[0] = self.interest_rate
        self.went_bankrupt_hist[0] = self.went_bankrupt
        self.system_money_spent_hist[0] = self.system_money_spent

    
    def _employment_probability(self) -> np.ndarray:
        """Probability for each company to get a worker. An additional "ghost company" is added to reflect unemployment. 

        Returns:
            np.ndarray:: self.N sized array with probabilities for each company to get a worker.
        """
        if self.salary.sum() != 0:
            prob_get_a_worker = self.salary / (self.salary).sum()
        else: 
            prob_get_a_worker = np.ones(self.N) / self.N
        return prob_get_a_worker


    def _redistribute_workers(self):
        self.w[:] = 0  # "Fire" all workers
        
        # All workers picks a company to work for with some probability
        idx_each_worker_choose = np.random.choice(np.arange(self.N), size=self.W, replace=True, p=self._employment_probability())  # idx of companies that each worker chooses to work for
        idx_company_that_hires, number_of_workers_hired = np.unique(idx_each_worker_choose, return_counts=True)

        # Update values
        self.w[idx_company_that_hires] = number_of_workers_hired
        

    def _sell_and_salary(self):
        # Pick the indices of the companies that will sell
        idx_companies_selling = np.random.choice(np.arange(self.N), size=self.N, replace=True)
        # Find which companies that sells and how many times they do it
        idx_unique, counts = np.unique(idx_companies_selling, return_counts=True)
        # Find how much each of the chosen companies sell for and how much they pay in salary
        sell_unique = self.w[idx_unique] * self.system_money_spent_last_step / self.W  
        salary_unique = self.salary[idx_unique] * self.w[idx_unique]
        # Multiply that by how many times they sold and paid salaries
        sell_unique_all = sell_unique * counts
        salary_unique_all = salary_unique * counts
        
        # Update values
        self.d[idx_unique] += salary_unique_all - sell_unique_all
        self.system_money_spent += salary_unique_all.sum()
        

    def _pay_interest(self) -> None:   
        positive_debt_idx = self.d > 0
        money_spent_on_interest = self.d[positive_debt_idx] * self.interest_rate
        self.d[positive_debt_idx] += money_spent_on_interest
        # self.system_money_spent += money_spent_on_interest.sum()
    

    def _update_salary(self, time_step):
        """Each company changes its salary. 
        If a company expects to loose workers such that w < m, it will increase salary. Otherwise it decreases it.
        """
        # Values for increased and decreased salary
        noise_factor = np.random.uniform(0, 1, size=self.N)
        salary_increase_with_noise = self.salary_increase * noise_factor
        negative_correction_factor = salary_increase_with_noise / (1 - salary_increase_with_noise)
        
        increased_salary_val = self.salary * (1 + negative_correction_factor)
        decreased_salary_val = self.salary * (1 - salary_increase_with_noise) 
        
        # Find who wants to increase 
        delta_debt = self.d - self.d_hist[:, time_step - 1]
        companies_want_to_increase_salary = delta_debt <= 0        
        
        # Make update and set minimum salary
        self.salary = np.where(companies_want_to_increase_salary, increased_salary_val, decreased_salary_val)
        self.salary = np.maximum(self.salary, self.salary_min)  # self.salary_min


    def _bankruptcy(self):
        """Companies who pays more in debt than they earn in sellings goes bankrupt.
        """
        # Find companies that go bankrupt
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
            # self.salary[bankrupt_idx] = self.salary[new_salary_idx] * 1 + np.random.uniform(-self.salary_increase, self.salary_increase, size=number_of_companies_gone_bankrupt)
            self.salary[bankrupt_idx] = self.salary[new_salary_idx] * 1 + np.random.uniform(-0.01, 0.01, size=number_of_companies_gone_bankrupt)
        else:
            self.salary = np.random.uniform(self.salary_increase, 1, number_of_companies_gone_bankrupt)
        
        # Set minimum salary
        self.salary = np.maximum(self.salary, self.salary_min)  # self.salary_min
        
       
    def _time_scale_0(self, **kwargs):
        return 0
    
    
    def _time_scale_x(self, **kwargs):
        x = kwargs["x"]
        return x
    
    
    def _time_scale_inverse_r(self, **kwargs):
        return int(1 / self.interest_rate)
        
       
    def _probability_of_default(self, time_step, T) -> None:
        if time_step > T:
            bankrupt_T_ago_to_now = np.append(self.went_bankrupt_hist[time_step - T : time_step - 1], [self.went_bankrupt])
            self.PD = np.mean(bankrupt_T_ago_to_now) / self.N
            self.PD = np.minimum(self.PD, 0.99)  # Prevent division by zero.
        
    
    def _adjust_interest_for_default(self, time_step) -> None:
        # Using the probability of default (synonymous with bankruptcy) to adjust the interest rate
        self.time_scale = self._func_time_scale(**{"x": 50})
        self._probability_of_default(time_step, T=self.time_scale)
        self.interest_rate = (1 + self.interest_rate_free) / (1 - self.PD) - 1 
        
    
    def _store_values_in_hist_arrays(self, time_step: int) -> None:
        # Company variables
        self.w_hist[:, time_step] = self.w
        self.d_hist[:, time_step] = self.d
        self.salary_hist[:, time_step] = self.salary

        # System variables
        self.interest_rate_hist[time_step] = self.interest_rate
        self.went_bankrupt_hist[time_step] = self.went_bankrupt 
        self.system_money_spent_hist[time_step] = self.system_money_spent 
        
        # Reset values
        self.went_bankrupt = 0  # Reset for next time step
        self.system_money_spent_last_step = self.system_money_spent  # For next time step
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
        
        # If the exact filename already exists, open in write, otherwise in append mode
        with h5py.File(self.file_path, "a") as f:
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
            group.create_dataset("system_money_spent", data=self.system_money_spent_hist)
            
            # Attributes
            group.attrs["W"] = self.W
            group.attrs["salary_increase"] = self.salary_increase
            group.attrs["interest_rate_free"] = self.interest_rate_free
            group.attrs["rf_space"] = self.rf_space
            group.attrs["ds_space"] = self.ds_space


    def store_data_over_parameter_space(self):
        """Create data for different values of rho and store it in the same file as the simulation data.
        """
        # Fix the seed such that the parameter is the only thing changing
        random_seed = np.random.randint(0, 100000)
        np.random.seed(random_seed)
        
        # Depending on whether rho or machine cost is investigated, change the variable
        N_sim = len(self.ds_space)
        for i, rho in enumerate(self.ds_space):
            print(f"{i+1}/{N_sim}")
            self.salary_increase = rho
            self.group_name = self._get_group_name()
            self.store_data()
            
    
    def store_data_s_min(self):
        salary_min_list = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]       
        for s_min in salary_min_list:
            self.salary_min = s_min
            self.group_name = self._get_group_name() + f"_smin{s_min}"
            self.store_data()
            
    
    def store_data_rf_and_ds(self):        
        total_iterations = len(self.ds_space) * len(self.rf_space)
        
        for i, ds in enumerate(self.ds_space):
            self.salary_increase = ds
            for j, r_f in enumerate(self.rf_space):
                # Get data
                self.interest_rate_free = r_f
                self.group_name = self._get_group_name()
                self.store_data()
                
                print(f"{i*len(self.rf_space) + j + 1}/{total_iterations}")


    def store_data_time_scale(self):
        func_time_scale_list = [self._time_scale_0, self._time_scale_x, self._time_scale_inverse_r]
        for func_time_scale in func_time_scale_list:
            self._func_time_scale = func_time_scale
            self.group_name =  self._get_group_name() + f"{func_time_scale.__name__}"
            self.store_data()
        
            
# Define variables for other files to use
number_of_companies = 250
number_of_workers = 5000
time_steps = 20_000
interest_rate_free = 0.05
salary_increase = 0.05
ds_space = np.linspace(0.01, 0.06, 15)  # Good range: 0.008 < ds < 0.08 
rf_space = np.linspace(0.01, 0.1, len(ds_space))  # Good range: 0.01 < rf < ?

seed = None

# Other files need some variables
workforce = Workforce(number_of_companies, number_of_workers, salary_increase, interest_rate_free, time_steps, ds_space, rf_space, seed)

dir_path_output = workforce.dir_path_output
dir_path_image = workforce.dir_path_image
group_name = workforce.group_name

if __name__ == "__main__":
    # workforce.store_data()
    
    # workforce.store_data_over_parameter_space()
    # workforce.store_data_s_min()
    workforce.store_data_rf_and_ds()
    # workforce.store_data_time_scale()
    print("Stored Values")