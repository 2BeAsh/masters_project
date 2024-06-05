import numpy as np
import matplotlib.pyplot as plt
import general_functions
from tqdm import tqdm
import networkx as nx
from time import time

import matplotlib.animation as animation
import functools


class DebtDeflation():
    
    def __init__(self, N: int, time_steps: int, interest: float, K_market: float, K_rival: float):
        self.N = N
        self.time_steps = time_steps
        self.interest = interest
        self.K_market = K_market
        self.K_rival = K_rival
        self.dir_path = "code/"
        self.dir_path_output = self.dir_path + "output/"
        self.dir_path_image = self.dir_path + "image/"
        self.file_parameter_addon = f"_steps{self.time_steps}_N{self.N}_interest{self.interest}_Kmarket{self.K_market}_Krival{self.K_rival}"
        
    
    def _initial_market(self) -> None:
        self.company_value = np.ones(self.N)
        self.debt = np.zeros(self.N)
        self.money = np.zeros(self.N)


    def _create_rival_network(self, connection_probability=0.1) -> None:
        """Initialize Erdos-Renyi network for who you are rivals with.
        """
        self.rival_network = nx.erdos_renyi_graph(n=self.N, p=connection_probability)        


    def _target_market_sigmoidal_relation(self, target_idx: int) -> float:
        """Relationship between one company and the rest of the market. Market add, rivals subtract.

        Args:
            target_idx (int): Target company.

        Returns:
            float: Term added to the change in money due to company-market relation
        """
        # Sum up all of the market, excluding the target and rivals
        # Get rival indices
        rival_idx = list(self.rival_network.neighbors(target_idx))
        companies_to_exclude = rival_idx + [target_idx]
        company_value_excluding = np.delete(self.company_value, companies_to_exclude)
        # company_value_excluding_target = self.company_value[np.arange(self.N) != target_idx]
        market_sum_excluding = np.sum(company_value_excluding)
        
        # Target-market relation
        target_market_relation = self.company_value[target_idx] * market_sum_excluding / (self.K_market + market_sum_excluding)
        
        # Target-rival relation
        rival_sum = np.sum(self.company_value[rival_idx])
        target_rival_relation = - rival_sum * K_rival  # NOTE should probably change to non-linear
        
        return target_market_relation + target_rival_relation
        
        
    def _target_market_square(self, target_idx):
        company_value_excluding = np.delete(self.company_value, [target_idx])
        # company_value_excluding_target = self.company_value[np.arange(self.N) != target_idx]
        market_sum_excluding = np.sum(company_value_excluding)
        market = market_sum_excluding - (market_sum_excluding / self.K_market) ** 2
        
        return self.company_value[target_idx] * market
    

    def _delta_money(self, target_idx: int) -> None:
        target_market_relation = self._target_market_square(target_idx)
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
        # Initialize market and network
        self._initial_market()
        self._create_rival_network()
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
            # NOTE calculates the same things (e.g. delta_debt) many times. Could be optimized
            
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
    
    
    def _load_data(self, filename):
        if filename == "": 
            filename = self.dir_path_output + "debt_deflation_steps" + self.file_parameter_addon + ".npy"
        data_all = np.load(filename)
        company_value = data_all[:, :, 0]
        debt = data_all[:, :, 1]
        money = data_all[:, :, 2]
        
        return company_value, debt, money
    
        
    def plot_means(self, filename=""):
        """Plot averages.
        """
        # Load data and create time values array
        company_value, debt, money = self._load_data(filename)
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
        display_parameters_str = (r"$N_{agent} = $" + str(self.N) 
                                  + "\n" + r"Interest $=$ " + str(self.interest) 
                                  + "\n" + r"$K_{market} = $ " + str(self.K_market)
                                  + "\n" + r"$K_{rival} = $ " + str(self.K_rival))
        ax.text(x=0.01, y=0.9, s=display_parameters_str, transform=ax.transAxes, horizontalalignment='left', verticalalignment='center')
        # Save figure
        figname = self.dir_path_image + f"means" + self.file_parameter_addon + ".png"
        plt.savefig(figname)
        plt.show()


    def plot_means_5(self, filename=""):
        """Plot averages.
        """
        # Load data and create time values array
        company_value, debt, money = self._load_data(filename)
        N_firma = 99
        company_value_5 = company_value[:N_firma, :].T
        debt_5 = debt[:N_firma, :].T
        money_5 = money[:N_firma, :].T
        
        # Averages
        company_value_mean = np.mean(company_value, axis=0)
        debt_mean = np.mean(debt, axis=0)
        money_mean = np.mean(money, axis=0)
        time_values = np.arange(0, self.time_steps)

        # Plot averages single axis
        fig, ax = plt.subplots()
        # ax.plot(time_values, company_value_mean, label="Company value")
        # ax.plot(time_values, debt_mean, label="Debt")
        # ax.plot(time_values, money_mean, label="Money")

        ax.plot(time_values, company_value_5, label="x", c="black")
        ax.plot(time_values, debt_5, label="D", c="red")
        ax.plot(time_values, money_5, label="M", c="blue")

        # Figure setup
        ax.set(xlabel="Time", ylabel="Value", yscale="log")
        # ax.legend(ncols=3, bbox_to_anchor=(0.5, 0.95), loc="lower center")
        # Text
        display_parameters_str = (r"$N_{agent} = $" + str(self.N) 
                                  + "\n" + r"Interest $=$ " + str(self.interest) 
                                  + "\n" + r"$K_{market} = $ " + str(self.K_market)
                                  + "\n" + r"$K_{rival} = $ " + str(self.K_rival))
        ax.text(x=0.01, y=0.9, s=display_parameters_str, transform=ax.transAxes, horizontalalignment='left', verticalalignment='center')
        # Save figure
        figname = self.dir_path_image + f"means" + self.file_parameter_addon + ".png"
        plt.savefig(figname)
        plt.show()

    
    
    def animate_size_distribution(self, filename=""):
        time_i = time()
        # Load data and create time values array
        company_value, debt, money = self._load_data(filename)
        # Bin data
        Nbins = int(np.sqrt(self.time_steps))
        bin_edges = np.linspace(company_value.min(), company_value.max(), Nbins)
        
        fig, ax = plt.subplots()
        # n, _ = np.histogram(company_value[:, 0], bin_edges)  
        _, _, bar_container = ax.hist(company_value[:, 0], bin_edges)  # Initial histogram 
        ax.set(xlim=(bin_edges[0], bin_edges[-1]), title="Time = 0")
        # Text
        display_parameters_str = (r"$N_{agent} = $" + str(self.N) 
                                  + "\n" + r"Interest $=$ " + str(self.interest) 
                                  + "\n" + r"$K_{market} = $ " + str(self.K_market)
                                  + "\n" + r"$K_{rival} = $ " + str(self.K_rival))
        ax.text(x=0.8, y=0.9, s=display_parameters_str, transform=ax.transAxes, horizontalalignment='left', verticalalignment='center')
  
        def animate(i, bar_container):
            data = company_value[:, i]
            n, _ = np.histogram(data, bin_edges)
            for count, rect in zip(n, bar_container.patches):
                rect.set_height(count)
            
            ax.set_title(f"Time = {i}")
            return bar_container.patches
        
        anim = functools.partial(animate, bar_container=bar_container)
        ani = animation.FuncAnimation(fig, anim, frames=self.time_steps, interval=1)
        
        time_create_ani = time()
        
        animation_name = self.dir_path_image + "size_distribution_animation" + self.file_parameter_addon + ".mp4"
        ani.save(animation_name, fps=30)
        
        time_save_ani = time()
        print("Time creating animation: \t", time_create_ani - time_i)
        print("Time saving animation: \t", time_save_ani - time_create_ani)
        

if __name__ == "__main__":
    # Parameters
    N_agents = 100
    time_steps = 1000
    interest = 1.25
    K_market = 10 * N_agents
    K_rival = 0
    
    debtdeflation = DebtDeflation(N=N_agents, time_steps=time_steps, interest=interest, K_market=K_market, K_rival=K_rival)
    filename = debtdeflation.dir_path_output + "debt_deflation_steps" + f"_steps{time_steps}_N{N_agents}_interest{interest}_Kmarket{K_market}_Krival{K_rival}" + ".npy"
    
    generate_data = True
    if generate_data == True:
        debtdeflation.simulation()
    
    debtdeflation.plot_means_5(filename)
    # debtdeflation.animate_size_distribution()
    