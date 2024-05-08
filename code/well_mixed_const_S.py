# Well mixed
# Fixed fraction suscpetible
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
from master import attempt_conversion, rng, non_S_val, S_val, E_val, I_val, dir_path


# -- Parameters --
dir_path_image = dir_path + "image/well_mixed_const_S_"
dir_path_output = dir_path + "output/well_mixed_const_S_"


def initial_population(N_tot: int, N_S: int, N_I: int):
    """The population at time 0. 

    Args:
        N_tot (int): Population size
        N_S (int): People who are susceptible to conversion
        N_I (int): Initial cultists. 

    Returns:
        array: Initial population with randomly chosen cultists and susceptible
    """
    population = non_S_val * np.ones(N_tot)
    population_possible_indices = np.arange(N_tot)
    # Get indices for S and I
    idx_S = rng.choice(a=population_possible_indices, size=N_S, replace=False)  # Draw N_S indices from population

    idx_chosen_by_S = np.in1d(population_possible_indices, idx_S)
    population_possible_indices_I = population_possible_indices[~idx_chosen_by_S]
    idx_I = rng.choice(a=population_possible_indices_I, size=N_I, replace=False)  # Cannot draw the members that S drew
    # Update values
    population[idx_S] = S_val
    population[idx_I] = I_val
    return population


def evolve(N_tot: int, N_S: int, N_I: int, time_steps: int, dir_path_file=dir_path_output):    
    """Time evolution of the cult conversion. 

    Args:
        N_tot (int): Population size
        N_S (int): Number of susceptible people.
        N_I (int): Number of cultits at time = 0.
        time_steps (int): How long the code is run
        dir_path_file (str, optional): Path to save the image to. Defaults to dir_path_output.
    """
    population = initial_population(N_tot, N_S, N_I)
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
        # Store current population in history
        population_history[:, i] = 1 * population  # Need a copy
    
    # Write to file
    np.savetxt(fname=dir_path_file + "population_history.gz", X=population_history)
    

def plot_SIE():
    """Plot the Susceptible, Exposed and Cultist over time. 
    """
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
    N_tot = 1000
    frac_S = 0.8
    frac_I = 0.1
    time_steps = 35_000
    
    get_data = True
    if get_data:
        evolve(N_tot, int(N_tot*frac_S), int(N_tot*frac_I), time_steps)
    
    plot_SIE()