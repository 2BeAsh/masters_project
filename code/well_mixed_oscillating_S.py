# Well mixed
# All susceptible oscillate between Non-S and S
# At each time step, pick three people. If...
    # 2 are I, 1 is S, S -> E
    # 1 is I, >1 is E, one of E->I
    # Otherwise skip

# Number explanation:
    # -1 == Non-S (Cannot be converted)
    # 0 == S (Susceptible)
    # 1 == E (Exposed)
    # 2 == I (Cultist)
    
    
# -- Imports --
import numpy as np
import matplotlib.pyplot as plt
import general_functions as gf
from tqdm import tqdm
from master import update_who_is_susceptible, attempt_conversion, rng, non_S_val, S_val, E_val, I_val, dir_path


# -- Parameters --
name = "well_mixed_oscillation_S_"
dir_path_image = dir_path + "image/" + name
dir_path_output = dir_path + "output/" + name


def initial_population(N_tot: int, p_S: int, N_I: int):
    """The population at time 0. 

    Args:
        N_tot (int): Population size
        p_S (float): Probability for a non-S to be converted to S. Pr(non-S -> S).
        N_I (int): Number of initial cultists. 

    Returns:
        array: Initial population with randomly chosen cultists and susceptible
    """
    population = non_S_val * np.ones(N_tot)
    population_possible_indices = np.arange(N_tot)
    # Get indices for S and I
    N_S = int(p_S * N_tot)  # Number of people initially susceptible
    print("Initial number of Susceptible = ", N_S, " out of total population = ", N_tot)
    idx_S = rng.choice(a=population_possible_indices, size=N_S, replace=False)  # Draw N_S indices from population
    
    # Not chosen by S
    idx_chosen_by_S = np.in1d(population_possible_indices, idx_S)
    population_possible_indices_I = population_possible_indices[~idx_chosen_by_S]
    idx_I = rng.choice(a=population_possible_indices_I, size=N_I, replace=False)  # Same, but cannot draw same as S
    # Update values
    population[idx_S] = S_val
    population[idx_I] = I_val
    return population


def evolve(N_tot, p_S, p_non_S, N_I, time_steps, dir_path_file=dir_path_output):    
    """Time evolution of the cult conversion. 

    Args:
        N_tot (int): Population size
        N_s (int): Number of susceptible people.
        N_I (int): Number of cultits at time = 0.
        time_steps (int): How long the code is run
        dir_path_file (str, optional): Path to save the image to. Defaults to dir_path_output.
    """
    population = initial_population(N_tot, p_S, N_I)
    population_history = np.empty((N_tot, time_steps))
    population_history[:, 0] = 1 * population
    for i in tqdm(range(1, time_steps)):
        # Pick three people, one target and two conversation initialisers. 
        # Then get their states
        people_picked_idx = rng.choice(np.arange(N_tot), size=3, replace=False)
        target_idx, init1_idx, init2_idx = people_picked_idx
        target_state = population[target_idx]
        init1_state = population[init1_idx]
        init2_state = population[init2_idx]
        
        # Perform conversion attempt and update target's state
        target_new_state = attempt_conversion(target_state, init1_state, init2_state)
        population[target_idx] = target_new_state
        
        # Check if changes between susceptible and non susceptible.
        population = update_who_is_susceptible(population, p_S, p_non_S)
        
        # Store current population in history
        population_history[:, i] = 1 * population  # Need a copy
    
    # Write to file
    np.savetxt(fname=dir_path_file + "population_history.gz", X=population_history)
    

def plot_SIE():
    # Load data
    population_history = np.genfromtxt(dir_path_output+"population_history.gz")
    N_tot, time = np.shape(population_history)
    S_history = np.count_nonzero(population_history==S_val, axis=0)
    E_history = np.count_nonzero(population_history==E_val, axis=0)
    I_history = np.count_nonzero(population_history==I_val, axis=0)
    
    # Normalize
    S_history = S_history / N_tot
    E_history = E_history / N_tot
    I_history = I_history / N_tot
    # Plot
    fig, ax = plt.subplots()
    time_vals = np.arange(time)
    ax.plot(time_vals, S_history, "-", label="Susceptible")
    ax.plot(time_vals, E_history, "--", label="Exposed")
    ax.plot(time_vals, I_history, "-.", label="Cultist")
    ax.legend(bbox_to_anchor=(0.5, 1.05), ncol=3, loc="lower center")
    figname = dir_path_image + "SIE_plot.png"
    plt.savefig(figname)
    plt.show()
    
    
if __name__ == "__main__":
    # Parameters
    N_tot = 1000
    N_I = 100
    p_S = 0.01  # P(non-S -> S)
    p_non_S = 0.05  # P(S -> non_S)
    
    time_steps = 35_0
    
    # Run evolution
    get_data = True
    if get_data:
        evolve(N_tot, p_S, p_non_S, N_I, time_steps)
    
    # Visualize
    plot_SIE()