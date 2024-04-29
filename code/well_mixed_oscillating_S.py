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
plt.style.use("code/presentation.mplstyle")

# -- Parameters --
rng = np.random.default_rng()
non_S_val = -1
S_val = 0
E_val = 1
I_val = 2
dir_path = "code/"
dir_path_image = dir_path + "image/well_mixed_oscillation_S_"
dir_path_output = dir_path + "output/well_mixed_oscillation_S_"


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

    
def attempt_conversion(target: int, init1: int, init2: int) -> int:
    """Given the state of three people, one target and two conversation initialisers, check if a conversion happens.

    Args:
        target (int): State of the possible convertee.
        init1 (int): State of the first conversation initialiser.
        init2 (int): State of the second conversation initialiser.

    Returns:
        int: target's updated value. 
    """
    # Target cannot be I
    # Target is Susceptible, both initialisers are cultists. S - > E
    if (target == S_val) and (init1 == I_val) and (init2 == I_val):
        target = E_val
    # Target is Exposed, at least one initialiser is cultist. E - > I
    elif (target == E_val) and ((init1 == I_val) or (init2 == I_val)):
        target = I_val
    return target


def update_who_is_susceptible(population, p_S: float, p_non_S: float):
    """At each time step, susceptible and non-susceptible can change to the other state. 
    Get indices, draw a random number for each and compare to probability and update value accordingly

    Args:
        population (array): Population state array.
        p_S (float): Pr(non-S -> S).
        p_non_S (float): Pr(S -> non-S).

    Returns:
        array: Updated population state array
    """
    population = 1 * population
    # S -> non-S
    S_idx = np.nonzero(population==S_val)[0]  # Idx of susceptible people
    rng1 = rng.uniform(low=0, high=1, size=S_idx.size)  # Random number for each person
    S_change = rng1 < p_non_S  # Check who changes to non-S
    S_change_population_idx = S_idx[S_change]  # Get population indices of those who change to non_S
    population[S_change_population_idx] = non_S_val  # Update population
    
    # non-S -> S
    non_S_idx = np.nonzero(population==non_S_val)[0]  # Get indicides of the non-Susceptible people
    rng2 = rng.uniform(low=0, high=1, size=non_S_idx.size)  # Draw random number for each
    non_S_change = rng2 < p_S  # Check who changes to S i.e. undergoes crisis
    non_S_change_population_idx = non_S_idx[non_S_change]  # Get their indicis in the population array
    population[non_S_change_population_idx] = S_val  # Update the population array
    
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