import numpy as np
import matplotlib.pyplot as plt
import general_functions
from tqdm import tqdm


class DebtDeflation():
    
    def __init__(self, N: int, time_steps: int, interest: float, K_market: float):
        self.N = N
        self.time_steps = time_steps
        self.interest = interest
        self.K_market = K_market
        self.dir_path = "code/"
        self.dir_path_output = self.dir_path + "output/"
        self.dir_path_image = self.dir_path + "image/"
        self.file_parameter_addon = f"_steps{self.time_steps}_N{self.N}_interest{self.interest}_Kmarket{self.K_market}"
        
    
    def _initial_market(self) -> None:
        self.company_value = np.ones(self.N)
        self.debt = np.zeros(self.N)
        self.money = np.zeros(self.N)


    def _target_market_sigmoidal_relation(self, target_idx: int) -> float:
        company_value_excluding_target = self.company_value[np.arange(self.N) != target_idx]
        market_sum_excluding_target = np.sum(company_value_excluding_target)
        return self.company_value[target_idx] * market_sum_excluding_target / (self.K_market + market_sum_excluding_target)


    def _delta_money(self, target_idx: int) -> None:
        target_market_relation = self._target_market_sigmoidal_relation(target_idx)
        self.delta_money = target_market_relation - self.interest * self.debt[target_idx]


    def _delta_debt(self) -> None:
        self.delta_debt = self.delta_money / self.interest
    
    
    def _delta_company_value(self) -> None:
        self.delta_company_value = self.delta_money + self.delta_debt


    def _step(self, target_idx: int) -> None:
        # Update steps
        self._delta_money(target_idx)
        self._delta_debt()
        self._delta_company_value()
        
        # Update values and check for bankruptcy
        self.company_value[target_idx] += self.delta_company_value
        if self.company_value[target_idx] > 0:  # Not bankrupt
            self.money[target_idx] += self.delta_money
            self.debt[target_idx] += self.delta_debt
        else: # Bankrupt, start new company
            self.money[target_idx] = 0
            self.debt[target_idx] = 0
            self.company_value[target_idx] = 1
            
    
    def simulation(self) -> None:
        # Initial values
        self._initial_market()
        # History
        company_value_hist = np.empty((self.N, self.time_steps))
        debt_hist = np.empty_like(company_value_hist)
        money_hist = np.empty_like(company_value_hist)
        
        company_value_hist[:, 0] = self.company_value
        debt_hist[:, 0] = self.debt
        money_hist[:, 0] = self.money
        
        # Time evolution
        for i in tqdm(range(1, self.time_steps)):
            # Pick random agent to update
            target_idx = np.random.randint(low=0, high=self.N)
            # Timestep update, then append values to history
            # NOTE calculates the same things (e.g. delta_debt) many times. 
            
            self._step(target_idx)  # Updates company_value, debt and money
            company_value_hist[:, i] = self.company_value
            debt_hist[:, i] = self.debt
            money_hist[:, i] = self.money
        
        # Store in file
        all_data = np.empty((self.N, self.time_steps, 3))
        all_data[:, :, 0] = company_value_hist
        all_data[:, :, 1] = debt_hist
        all_data[:, :, 2] = money_hist
        filename = self.dir_path_output + "debt_deflation_steps" + self.file_parameter_addon + ".npy"
        np.save(filename, arr=all_data)
        
        
    def plot(self):
        # Load data and create time values array
        filename = self.dir_path_output + "debt_deflation_steps" + self.file_parameter_addon + ".npy"
        data_all = np.load(filename)
        company_value = data_all[:, :, 0]
        debt = data_all[:, :, 1]
        money = data_all[:, :, 2]
        
        # Averages
        company_value_mean = np.mean(company_value, axis=0)
        debt_mean = np.mean(debt, axis=0)
        money_mean = np.mean(money, axis=0)
        time_values = np.arange(0, self.time_steps)

        # Plot averages single axis
        fig, ax = plt.subplots()
        ax.plot(time_values, company_value_mean, label="Company value")
        ax.plot(time_values, debt_mean, label="Debt")
        ax.plot(time_values, money_mean, label="Money")
        
        # Figure setup
        ax.set(xlabel="Time", ylabel="Mean Value")
        ax.legend(ncols=3, bbox_to_anchor=(0.5, 0.95), loc="lower center")
        # Text
        display_parameters_str = r"$N_{agent} = $" + str(self.N) + "\n" + r"Interest $=$ " + str(self.interest) + "\n" + r"$K_{market} = $ " + str(self.K_market)
        ax.text(x=0.01, y=0.9, s=display_parameters_str, transform=ax.transAxes, horizontalalignment='left', verticalalignment='center')
        # Save figure
        figname = self.dir_path_image + f"means" + self.file_parameter_addon + ".png"
        plt.savefig(figname)
        plt.show()
        

if __name__ == "__main__":
    # Parameters
    N_agents = 10_000
    time_steps = 1000
    interest = 1.25
    K_market = 10
    
    debtdeflation = DebtDeflation(N=N_agents, time_steps=time_steps, interest=interest, K_market=K_market)
    
    generate_data = True
    if generate_data == True:
        debtdeflation.simulation()
    
    debtdeflation.plot()
    