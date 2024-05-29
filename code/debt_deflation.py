import numpy as np
import matplotlib.pyplot as plt
import general_functions



class DebtDeflation():
    
    def __init__(self, N, time_steps):
        self.N = N
        self.time_steps = time_steps
        
    

    def _initial_market(self):
        self.company_value = np.ones(self.N)
        self.debt = np.zeros(self.N)
        self.money = np.zeros(self.N)


    def delta_company_value(self):
        
        return 


    def _delta_money(self):
        
        return 


    def _delta_debt(self):

        return 
        

    def _step(self):
        self.money += self._delta_money()
        self.debt += self._delta_debt()
        self.company_value += self.delta_company_value()


    def simulation(self):
        # Initial values
        self._initial_market()
        # History
        company_value_hist = np.empty((self.N, self.time_steps))
        debt_hist = np.empty_like(company_value_hist)
        money_hist = np.empty_like(company_value_hist)
        company_value_hist[:, 0] = self.company_value
        debt_hist[:, 0] = self.debt
        money_hist[:, 0] = self.money
        
        for i in range(1, self.time_steps):
            # Perform timestep update, then append values to history
            self._step()
            company_value_hist[:, i] = self.company_value
            debt_hist[:, i] = self.debt
            money_hist[:, i] = self.money
        
        # Store in file
        all_data = np.empty((self.N, self.time_steps, 3))
        all_data[:, :, 0] = company_value_hist
        all_data[:, :, 1] = debt_hist
        all_data[:, :, 2] = money_hist
        np.save(f"debt_deflation_steps{self.time_steps}.npy", arr=all_data)
        
        
    def plot(self):
        # Load data and create time values array
        data_all = np.load(f"debt_deflation_steps{self.time_steps}.npy")
        company_value = data_all[:, :, 0]
        debt = data_all[:, :, 0]
        money = data_all[:, :, 0]
        time_values = np.arange(0, self.time_steps)
        # Plot
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3)
        ax1.plot(time_values, company_value)
        
        ax2.plot(time_values, debt)
        
        ax3.plot(time_values, money)
        plt.show()