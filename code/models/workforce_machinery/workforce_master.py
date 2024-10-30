import numpy as np
from tqdm import tqdm
from pathlib import Path
import h5py


class Workforce():
    def __init__(self, number_of_companies, number_of_workers, interest_rate_change_size, salary_increase, time_steps):
        self.N = number_of_companies
        self.W = number_of_workers 
        self.rho = interest_rate_change_size
        self.salary_increase = salary_increase
        self.time_steps = time_steps
        
        # Local paths for saving files
        file_path = Path(__file__)
        self.dir_path = file_path.parent.parent.parent
        self.dir_path_output = Path.joinpath(self.dir_path, "output")
        self.dir_path_image = Path.joinpath(self.dir_path, "images", "image_workforce_machinery")
        self.dir_path_image.mkdir(parents=True, exist_ok=True)
        self.group_name = f"Steps{self.time_steps}_N{self.N}_W{self.W}"
        
        # Seed
        # np.random.seed(42)   


    def _initialize_market_variables(self) -> None:    
        # System variables
        self.interest_rate_free = 0.05
        self.interest_rate = self.interest_rate_free
        self.time_scale = 12
        self.machinery_price = self.time_scale * self.W / self.N  # N / W is average workers per company i.e. roughly income. OBS maybe should just be W / N, as income normally approximately equal to salary expenses?

        # Company variables
        self.w = np.ones(self.N, dtype=int)  # Initial workers
        self.m = np.ones(self.N, dtype=int)  # Initial machinery
        self.d = self.machinery_price * np.ones(self.N, dtype=float)  # All starts with debt equal to machinery price.        
        self.salary = np.random.uniform(0.01, 1.1, self.N)  # Pick random salaries
        
        # Worker variables
        self.unemployed = self.W - self.N  # Every company starts with one employee, N = sum w        
        
        # Initial values
        self.PD = 0
        self.went_bankrupt = 0


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


    def _employed(self):
        return self.W - self.unemployed
    
    
    def _production_capacity(self):
        return np.minimum(self.w, self.m)


    def _employment(self):
        # Update who wants to hire or buy machinery
        self._want_to_hire_or_buy_machinery()
        
        # Find the number of people able to be hired,
        # which is the minimum of unemployed and number companies that want to hire
        able_to_be_hired = int(np.clip(self.unemployed, 0, np.sum(self.want_to_hire)))
        
        # If no people are wanted or no people are unemployed, skip the employment process
        if able_to_be_hired == 0:
            return

        # Get salaries of the companies that want to employ.
        # Order the salaries of companies that want to employ,
        # and get the indices of the able_to_be_hired companies with the largest salaries.
        # Each of the top able_to_be_hired salary companies employ one worker.
        salary_of_companies_that_want_to_employ = self.salary[self.want_to_hire] 
        idx_ordered_salary_of_company_that_want_to_employ = np.argsort(salary_of_companies_that_want_to_employ)
        idx_companies_that_employ = idx_ordered_salary_of_company_that_want_to_employ[-able_to_be_hired:] 
        
        # Employ - Update values
        self.w[idx_companies_that_employ] = self.w[idx_companies_that_employ] + 1
        self.unemployed -= able_to_be_hired
        
        
    def _buy_machinery(self):
        # Update who wants to hire or buy machinery
        self._want_to_hire_or_buy_machinery()
        
        # All companies that want machinery buy machinery
        self.m[self.want_machinery] += 1
        self.d[self.want_machinery] -= self.machinery_price
        

    def _sell_and_salary(self):
        """Choose N random companies to sell. Whenever you sell, you also pay salaries.
        """
        # Choose random company to gain money from selling to employed people, but also to pay salaries
        for _ in range(self.N):
            idx = np.random.randint(0, self.N)
            # print(self._employed(), self.salary[idx], self.w[idx], self._production_capacity()[idx])
            self.d[idx] = self._employed() / self.W * (self.salary[idx] * self.w[idx] - self._production_capacity()[idx])


    def _pay_interest(self) -> None:   
        positive_debt_idx = self.d > 0
        self.d[positive_debt_idx] *= 1 + self.interest_rate
    

    def _quit_job(self):
        """All workers check compare their salary to that of a random company, quit if the other salary is better.
        """
        # Each company's workers check if they want to quit
        for i in range(self.N):
            # Get indices of all companies except the current one
            idx = np.arange(self.N, dtype=int)
            idx = idx[idx != i]
            # For each worker in the company, find another company to compare with
            idx_other_comp = np.random.choice(idx, size=self.w[i], replace=True)
            # Find the salary of the other company and compare
            salary_other_comp = self.salary[idx_other_comp]
            number_of_workers_who_quit = np.count_nonzero(salary_other_comp > self.salary[i])
            # Update values
            self.w[i] -= number_of_workers_who_quit
            self.unemployed += number_of_workers_who_quit


    def _want_to_hire_or_buy_machinery(self):
        """Companies want to hire or buy machinery if one of the two is limiting their growth.
            If the two are equal, buy machinery, as otherwise no growth will happen.
        """
        self.want_to_hire = self.w < self.m
        self.want_machinery = ~self.want_to_hire  # If does not want to hire, wants to buy machinery
        

    def _update_salary(self):
        """Each company changes its salary. 
        If the company wants to hire and did not hire last time step, increase salary.
        If the company does not want to hire, decrease salary
        """
        # Update if wants to hire or buy machinery
        self._want_to_hire_or_buy_machinery()
        
        # Increase salary by percent if want to hire 
        # Decrease salary if does not want to hire
        negative_correction_factor = 1 / (1 - self.salary_increase)
        
        increased_salary_val = self.salary * (1 + self.salary_increase * negative_correction_factor)
        decreased_salary_val = self.salary * (1 - self.salary_increase)
        
        self.salary = np.where(self.want_to_hire, increased_salary_val, decreased_salary_val)


    def _bankruptcy(self):
        # Goes bankrupt if min(w, m) < rd 
        bankrupt_idx = self._production_capacity() < self.interest_rate * self.d
        number_of_companies_gone_bankrupt = bankrupt_idx.sum()  # True = 1, False = 0, so sum gives amount of bankrupt companies
        workers_fired = self.w[bankrupt_idx].sum() - number_of_companies_gone_bankrupt  # Number of workers fired. Second term because new companies have 1 worker
        workers_fired = np.maximum(workers_fired, 0)  # Prevent negative workers fired. Can happen if a w = 0 company goes bankrupt.
        
        assert workers_fired >= 0, f"Number of workers fired is negative"

        # System values
        self.unemployed += workers_fired
        self.went_bankrupt = number_of_companies_gone_bankrupt  
                
        # Company values
        self.w[bankrupt_idx] = 1  # New company no workers
        self.d[bankrupt_idx] = self.machinery_price  # New company buys 1 machinery and gains debt equal to machinery price
        self.m[bankrupt_idx] = 1  # New company has 1 machinery
        
        # Pick salary of non-bankrupt companies and mutate it
        idx_surving_companies = np.arange(self.N)[~bankrupt_idx]
        new_salary_idx = np.random.choice(idx_surving_companies, size=np.sum(bankrupt_idx), replace=True)
        self.salary[bankrupt_idx] = self.salary[new_salary_idx] + np.random.uniform(-0.1, 0.1, np.sum(bankrupt_idx))
        
        # Set minimum salary
        self.salary = np.clip(self.salary, 0.01, 1.5)
        
       
    def _probability_of_default(self, time_step, T=12):
        if time_step > T + 1:
            self.PD = np.mean(self.went_bankrupt_hist[time_step - T - 1 : time_step - 1]) / self.N
            self.PD = np.minimum(self.PD, 0.99)  # Prevent division by zero.
        
    
    def _adjust_interest_for_default(self, time_step) -> None:
        # Using the probability of default (synonymous with bankruptcy) to adjust the interest rate
        self._probability_of_default(time_step)
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
            self._pay_interest()
            self._quit_job() 
            self._update_salary()
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
        file_name = "workforce_machinery_simulation_data.h5"
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
number_of_companies = 400
number_of_workers = 1500
time_steps = 500
interest_rate_change_size = 0.05  # rho, percentage change in r
salary_increase = 0.1

# Other files need some variables
workforce = Workforce(number_of_companies, number_of_workers, interest_rate_change_size, salary_increase, time_steps)

dir_path_output = workforce.dir_path_output
dir_path_image = workforce.dir_path_image
group_name = workforce.group_name

if __name__ == "__main__":
    print("You ran the wrong script :)")