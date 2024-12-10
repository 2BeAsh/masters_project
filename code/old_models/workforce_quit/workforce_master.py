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
        self.dir_path_image = Path.joinpath(self.dir_path, "images", "image_workforce_quit")
        self.dir_path_image.mkdir(parents=True, exist_ok=True)
        self.group_name = f"Steps{self.time_steps}_N{self.N}_W{self.W}"
        
        # Seed
        # np.random.seed(42)   


    def _initialize_market_variables(self) -> None:    
        # Company variables
        self.p = np.ones(self.N, dtype=float)
        self.d = np.zeros(self.N, dtype=float)
        self.want_to_hire = np.full(self.N, True, dtype=bool)
        self.salary = np.random.uniform(0.1, 1.5, self.N)  # Pick random salaries
        
        # Worker variables
        self.unemployed = self.W - self.N  # Every company starts with one employee
        
        # Bank variables
        self.interest_rate_free = 0.05
        self.interest_rate = self.interest_rate_free
        self.PD = 0
        self.went_bankrupt = 0


    def _initialize_hist_arrays(self) -> None:
        # Initialize hist arrays
        # Company
        self.p_hist = np.zeros((self.N, self.time_steps))
        self.d_hist = np.zeros((self.N, self.time_steps))
        self.salary_hist = np.zeros((self.N, self.time_steps)) 
        
        self.interest_rate_hist = np.zeros(self.time_steps, dtype=float)
        self.went_bankrupt_hist = np.zeros(self.time_steps, dtype=int)
        self.unemployed_hist = np.zeros(self.time_steps, dtype=int)
        
        # Initial values
        self.p_hist[:, 0] = self.p
        self.interest_rate_hist[0] = self.interest_rate * 1


    def _employed(self):
        return self.W - self.unemployed


    def _employment(self):
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
        self.p[idx_companies_that_employ] = self.p[idx_companies_that_employ] + 1
        self.unemployed -= able_to_be_hired
        

    def _sell(self):
        for _ in range(self.N):
            # Choose random company to gain money from selling to employed people
            idx = np.random.randint(0, self.N)
            self.d[idx] -= self.p[idx] * self._employed() / self.W


    def _pay_interest(self) -> None:   
        positive_debt_idx = self.d > 0
        self.d[positive_debt_idx] *= 1 + self.interest_rate
    
    
    def _pay_salaries(self):
        self.d += self.salary * self.p * self._employed() / self.W
    

    def _quit_job(self):
        """The probability that a company has a worker quit is higher if the salary is low and the company has many workers.
        Alternatively, for each company, check if each individual worker quits with prob (1 - salary) / W. 
        """
        # Find the companies where a worker quits
        random_numbers = np.random.uniform(0, 1, self.N)
        prob_quit = (1 - self.salary) * self.p / self.W
        companies_where_one_quit = np.where(random_numbers < prob_quit)
        
        # Update values
        self.p[companies_where_one_quit] = self.p[companies_where_one_quit] - 1
        self.unemployed += len(companies_where_one_quit[0])


    def _who_wants_to_hire(self, time_step):
        """Update the self.want_to_hire array based on the change in debt from the last time step."""
        # If a company made a profit the last time step, it wants to hire
        change_in_debt = self.d_hist[:, time_step - 1] - self.d_hist[:, time_step - 2]        
        self.want_to_hire = change_in_debt <= 0  # Reduced debt
        
        # Exception being potential start up companies, who always want to hire
        self.want_to_hire[self.p==0] = True
        

    def _update_salary(self):
        """Each company changes its salary. 
        If the company wants to hire and did not hire last time step, increase salary.
        If the company does not want to hire, decrease salary
        """
        # Increase salary by percent if want to hire 
        # Decrease salary if does not want to hire
        negative_correction_factor = 1 / (1 - self.salary_increase)
        
        increased_salary_val = self.salary * (1 + self.salary_increase * negative_correction_factor)
        decreased_salary_val = self.salary * (1 - self.salary_increase)
        
        self.salary = np.where(self.want_to_hire, increased_salary_val, decreased_salary_val)


    def _bankruptcy(self):
        # Goes bankrupt if p < rd  - OBS should it be p * w_e / w < r * d?
        bankrupt_idx = self.p < self.interest_rate * self.d
        workers_fired = self.p[bankrupt_idx].sum()  # Number of workers fired
        
        # Company values
        self.p[bankrupt_idx] = 0  # New company no workers
        self.d[bankrupt_idx] = 0  # New company no money/debt
        self.want_to_hire[bankrupt_idx] = True  # New company wants to hire

        # System values
        self.unemployed += workers_fired
        self.went_bankrupt = bankrupt_idx.sum()  # True = 1, False = 0, so sum gives amount of bankrupt companies

        # Pick salary of non-bankrupt companies and mutate it
        self.salary[bankrupt_idx] = np.random.uniform(0.1, 1.5, np.sum(bankrupt_idx))
        # idx_surving_companies = np.arange(self.N)[~bankrupt_idx]
        # new_salary_idx = np.random.choice(idx_surving_companies, size=np.sum(bankrupt_idx), replace=True)
        # self.salary[bankrupt_idx] = self.salary[new_salary_idx] + np.random.uniform(-0.1, 0.1, np.sum(bankrupt_idx))
        
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
        self.p_hist[:, time_step] = self.p
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
            self._sell()
            self._pay_interest()
            self._pay_salaries()
            self._quit_job()  # Should either be after salaries are pay or after salaries are updated
            self._who_wants_to_hire(time_step=i)
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
        file_name = "workforce_quit_simulation_data.h5"
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
        group.create_dataset("p", data=self.p_hist)
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
time_steps = 6000
interest_rate_change_size = 0.05  # rho, percentage change in r
salary_increase = 0.1

# Other files need some variables
workforce = Workforce(number_of_companies, number_of_workers, interest_rate_change_size, salary_increase, time_steps)

dir_path_output = workforce.dir_path_output
dir_path_image = workforce.dir_path_image
group_name = workforce.group_name

if __name__ == "__main__":
    print("You ran the wrong script :)")