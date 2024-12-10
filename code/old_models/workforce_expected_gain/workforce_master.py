import numpy as np
from tqdm import tqdm
from pathlib import Path
import h5py


class Workforce():
    def __init__(self, number_of_companies, number_of_workers, salary_increase, time_steps):
        self.N = number_of_companies
        self.W = number_of_workers 
        self.salary_increase = salary_increase
        self.time_steps = time_steps
        
        # Local paths for saving files
        file_path = Path(__file__)
        self.dir_path = file_path.parent.parent.parent
        self.dir_path_output = Path.joinpath(self.dir_path, "output")
        self.dir_path_image = Path.joinpath(self.dir_path, "images", "image_workforce_expected_gain")
        self.dir_path_image.mkdir(parents=True, exist_ok=True)
        self.group_name = f"Steps{self.time_steps}_N{self.N}_W{self.W}"
        
        # Seed
        # np.random.seed(42)   


    def _initialize_market_variables(self) -> None:    
        # System variables
        self.interest_rate_free = 0.05
        self.interest_rate = self.interest_rate_free
        self.time_scale = 24
        self.machinery_price = self.time_scale # self.W / self.N #self.time_scale * self.W / self.N  # N / W is average workers per company i.e. roughly income. OBS maybe should just be W / N, as income normally approximately equal to salary expenses?

        # Company variables
        self.w = int(self.W / (2 * self.N)) * np.ones(self.N, dtype=int)  # Initial workers
        self.m = np.ones(self.N, dtype=int)  # Initial machinery
        self.d = self.machinery_price * np.ones(self.N, dtype=float)  # All starts with debt equal to machinery price.        
        self.salary = np.random.uniform(0.01, 1.1, self.N)  # Pick random salaries
        
        # Worker variables
        self.unemployed = self.W - self.w.sum()  # Every company starts with one employee, N = sum w        
        
        # Initial values
        self.PD = 0
        self.went_bankrupt = 0
        
        # Other
        self.negative_correction_factor = 1 / (1 - self.salary_increase)


    def _initialize_hist_arrays(self) -> None:
        # Initialize hist arrays
        # Company
        self.w_hist = np.zeros((self.N, self.time_steps))
        self.m_hist = np.zeros((self.N, self.time_steps))
        self.d_hist = np.zeros((self.N, self.time_steps))
        self.salary_hist = np.zeros((self.N, self.time_steps)) 
        
        self.interest_rate_hist = np.zeros(self.time_steps, dtype=float)
        self.went_bankrupt_hist = np.zeros(self.time_steps, dtype=int)
        self.unemployed_hist = np.zeros(self.time_steps, dtype=int)
        
        # Initial values
        self.w_hist[:, 0] = self.w * 1
        self.m_hist[:, 0] = self.m * 1
        self.d_hist[:, 0] = self.d * 1
        self.salary_hist[:, 0] = self.salary * 1
        self.interest_rate_hist[0] = self.interest_rate * 1
        self.went_bankrupt_hist[0] = self.went_bankrupt * 1
        self.unemployed_hist[0] = self.unemployed * 1


    def _employed(self):
        return self.W - self.unemployed
    
    
    def _production_capacity(self):
        return np.minimum(self.w, self.m)
    
    
    def _expected_change_in_workers(self, input_salary):
        # Expected amount of workers hired
        expected_gain = self.unemployed * input_salary / self.salary.sum()
            
        # Expected amount of workers gained is fraction of companies with higher salary than what your salary is
        number_of_companies_where_salary_is_higher = np.count_nonzero(self.salary[:, None] > input_salary[None, :], axis=0)
        expected_loss = (self.w - 1) * number_of_companies_where_salary_is_higher / self.N
            
        return expected_gain - expected_loss
        
    
    def _expected_debt_change_from_increasing_salary(self):
        change_workers_decreased_salary = self._expected_change_in_workers(self.salary * (1 - self.salary_increase))
        change_workers_increased_salary = self._expected_change_in_workers(self.salary * (1 + self.salary * self.negative_correction_factor))
        
        debt_change_decrease_salary = (self.w + change_workers_decreased_salary) * (self.salary - 1)
        debt_change_increased_salary = (self.w + change_workers_increased_salary) * (self.salary * (1 + self.salary * self.negative_correction_factor) - 1)

        return debt_change_increased_salary - debt_change_decrease_salary


    def _employment(self):
        # Find the number of people able to be hired,
        # which is the minimum of unemployed and number companies that want to hire
        able_to_be_hired = np.maximum(self.unemployed, 0)
        
        # If no people are wanted or no people are unemployed, skip the employment process
        if able_to_be_hired == 0:
            return

        probability_to_employ_any_worker = self.salary / np.sum(self.salary)
        where_workers_was_employed = np.random.choice(a=np.arange(self.N), size=able_to_be_hired, replace=True, p=probability_to_employ_any_worker)
        idx_companies_that_employed, number_of_workers_employed = np.unique(where_workers_was_employed, return_counts=True)

        # Employ - Update values
        self.w[idx_companies_that_employed] = self.w[idx_companies_that_employed] + number_of_workers_employed
        self.unemployed -= able_to_be_hired
        
        
    def _buy_machinery(self):
        # Want to buy machinery if the expected number of workers is larger than the current machinery
        difference_in_expected_workers_and_current_machines = self.w + self._expected_change_in_workers(self.salary) - self.m
        machines_bought = np.maximum(difference_in_expected_workers_and_current_machines, 0).astype(np.int32)
        
        # Update values        
        self.m += machines_bought
        self.d += self.machinery_price * machines_bought
        

    def _sell_and_salary(self):
        """Choose N random companies to sell. Whenever you sell, you also pay salaries.
        """
        # Choose random company to gain money from selling to employed people, but also to pay salaries
        for _ in range(self.N):
            idx = np.random.randint(0, self.N)
            # print(self._employed(), self.salary[idx], self.w[idx], self._production_capacity()[idx])
            self.d[idx] += (self.salary[idx] * self.w[idx] - self._production_capacity()[idx]) * self._employed() / self.W


    def _pay_interest(self) -> None:   
        positive_debt_idx = self.d > 0
        self.d[positive_debt_idx] *= 1 + self.interest_rate
    

    def _quit_job(self):
        """All workers check compare their salary to that of a random company, quit if the other salary is better.
        """
        # Each company's workers, except 1, check if they want to quit
        for i in range(self.N):  
            # Get indices of all companies except the current one
            idx = np.arange(self.N, dtype=int)
            idx = idx[idx != i]
            # For each worker in the company, find another company to compare with
            idx_other_comp = np.random.choice(idx, size=self.w[i]-1, replace=True)  # -1 because one worker will never quit 
            # Find the salary of the other company and compare
            salary_other_comp = self.salary[idx_other_comp]
            number_of_workers_who_quit = np.count_nonzero(salary_other_comp > self.salary[i])
            # Update values
            self.w[i] -= number_of_workers_who_quit
            self.unemployed += number_of_workers_who_quit
        

    def _update_salary(self):
        """Each company changes its salary. 
        If a company expects to loose workers such that w < m, it will increase salary. Otherwise it decreases it.
        """
        # Values for increased and decreased salary
        increased_salary_val = self.salary * (1 + self.salary_increase * self.negative_correction_factor)
        decreased_salary_val = self.salary * (1 - self.salary_increase)
        
        # Find who wants to increase 
        companies_want_to_increase_salary = self._expected_debt_change_from_increasing_salary() < 0        
        
        # Make update
        self.salary = np.where(companies_want_to_increase_salary, increased_salary_val, decreased_salary_val)


    def _bankruptcy(self):
        # Goes bankrupt if min(w, m) < rd 
        # bankrupt_idx = self._production_capacity() < self.interest_rate * self.d
        bankrupt_idx = np.logical_or(self._production_capacity() < self.interest_rate * self.d, self.w == 0)
        number_of_companies_gone_bankrupt = bankrupt_idx.sum()  # True = 1, False = 0, so sum gives amount of bankrupt companies
        workers_fired = self.w[bankrupt_idx].sum()

        # System values
        self.unemployed += workers_fired - number_of_companies_gone_bankrupt 
        self.went_bankrupt = number_of_companies_gone_bankrupt  
        
        assert self.unemployed <= self.W, "Number of employed workers is larger than total number of workers" 
                
        # Company values
        self.w[bankrupt_idx] = 1  # New company one workers
        self.d[bankrupt_idx] = self.machinery_price  # New company buys 1 machinery and gains debt equal to machinery price
        self.m[bankrupt_idx] = 1  # New company has 1 machinery

        self.salary[bankrupt_idx] = np.random.uniform(0.01, 1.1, number_of_companies_gone_bankrupt)

        # # Pick salary of non-bankrupt companies and mutate it
        idx_surving_companies = np.arange(self.N)[~bankrupt_idx]
        if idx_surving_companies.size != 0:  # There are non-bankrupt companies            
            new_salary_idx = np.random.choice(idx_surving_companies, size=np.sum(bankrupt_idx), replace=True)
            self.salary[bankrupt_idx] = self.salary[new_salary_idx] + np.random.uniform(-0.1, 0.1, np.sum(bankrupt_idx))
        else:
            self.salary = np.random.uniform(0.01, 1.1, self.N)
        
        # Set minimum salary
        self.salary = np.maximum(self.salary, 0.01)
        
       
    def _probability_of_default(self, time_step, T) -> None:
        if time_step > T + 1:
            self.PD = np.mean(self.went_bankrupt_hist[time_step - T - 1 : time_step - 1]) / self.N
            self.PD = np.minimum(self.PD, 0.99)  # Prevent division by zero.
        
    
    def _adjust_interest_for_default(self, time_step) -> None:
        # Using the probability of default (synonymous with bankruptcy) to adjust the interest rate
        self._probability_of_default(time_step, T=self.time_scale)
        self.interest_rate = (1 + self.interest_rate_free) / (1 - self.PD) - 1 
        
    
    def _store_values_in_hist_arrays(self, time_step: int) -> None:
        # Company variables
        self.w_hist[:, time_step] = self.w
        self.m_hist[:, time_step] = self.m
        self.d_hist[:, time_step] = self.d
        self.salary_hist[:, time_step] = self.salary

        # System variables
        self.interest_rate_hist[time_step] = self.interest_rate * 1
        self.unemployed_hist[time_step] = self.unemployed * 1
        self.went_bankrupt_hist[time_step] = self.went_bankrupt * 1
        self.went_bankrupt = 0  # Reset for next time step
        
        
    def _simulation(self):
        # Initialize variables and hist arrays
        self._initialize_market_variables()
        self._initialize_hist_arrays()
        
        # Run simulation
        for i in tqdm(range(1, self.time_steps)):
            self._employment()
            self._buy_machinery()  # Should be after employment, because machinery is bought if m = w. 
            self._sell_and_salary()
            self._quit_job() 
            self._update_salary()
            self._pay_interest()
            self._bankruptcy()
            self._adjust_interest_for_default(time_step=i)
            self._store_values_in_hist_arrays(time_step=i)         
            
            # Check if number of unemployed and employed match total
            if self._employed() + self.unemployed != self.W:
                print("Number of employed and unemployed does not match total number of workers")
                # Print the values
                print(self._employed(), self.unemployed, self.W)
                # raise ValueError("Number of employed and unemployed does not match total number of workers")


    def store_values(self) -> None:
        # Check if output directory exists
        self.dir_path_output.mkdir(parents=True, exist_ok=True)
        
        # File name and path
        file_name = "workforce_expected_gain_simulation_data.h5"
        file_path = self.dir_path_output / file_name
        
        # If the exact filename already exists, open in write, otherwise in append mode
        f = h5py.File(file_path, "a")
        if self.group_name in list(f.keys()):
            f.close()
            f = h5py.File(file_path, "w")

        group = f.create_group(self.group_name)
        
        # Run simulation to get data
        self._simulation()
        
        # Store data in group
        # Company variables
        group.create_dataset("w", data=self.w_hist)
        group.create_dataset("m", data=self.m_hist)
        group.create_dataset("d", data=self.d_hist)
        group.create_dataset("s", data=self.salary_hist)
        
        # System variables
        group.create_dataset("interest_rate", data=self.interest_rate_hist)
        group.create_dataset("went_bankrupt", data=self.went_bankrupt_hist)
        group.create_dataset("unemployed", data=self.unemployed_hist)
        
        # Attributes
        group.attrs["W"] = self.W
        group.attrs["salary_increase"] = self.salary_increase
        f.close()

            
# Define variables for other files to use
number_of_companies = 300
number_of_workers = 2500
time_steps = 2000
salary_increase = 0.1

# Other files need some variables
workforce = Workforce(number_of_companies, number_of_workers, salary_increase, time_steps)

dir_path_output = workforce.dir_path_output
dir_path_image = workforce.dir_path_image
group_name = workforce.group_name

if __name__ == "__main__":
    print("You ran the wrong script :)")