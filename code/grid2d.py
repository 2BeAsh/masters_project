# Explanation
# 2d grid agent based
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


import numpy as np
import general_functions as gf
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm
from master import update_who_is_susceptible, rng, non_S_val, S_val, E_val, I_val, dir_path

# -- Parameters --
name = "grid_oscillation_S"
dir_path_image = dir_path + "image/" + name
dir_path_output = dir_path + "output/" + name


def conversion(converter, converted):
    # Pick person. If I, pulls one of its neighbours towards I (S -> E -> I)
    if converter != I_val:
        return converted
    elif converted == S_val:
        converted = E_val
    elif converted == E_val:
        converted = I_val
    return converted


def get_neighbour_idx(target_idx, array_size):
    """Given an index, return the index of one of its the four non-diagonal neighbours.

    Args:
        target_idx (int): Index for which the neighbour will be found
        array_size (int): Size of array

    Returns:
        int: Neighbour index
    """
    # Given an index, pick a random of its neighbours.
    # Check for out of bounds using array_size

    out_of_bounds = True  # Needed to run the loop at least once
    while out_of_bounds:
        neighbour_idx_relative_options = [(-1, 0), (0, -1), (1, 0), (0, 1)]
        neighbour_idx_relative = rng.choice(a=neighbour_idx_relative_options)
        neighbour_idx = target_idx + neighbour_idx_relative
        
        # Check for out of bounds
        lower_bound = (neighbour_idx < 0).any()
        upper_bound = (neighbour_idx >= array_size).any()  # Should not be >, only ==
        out_of_bounds = lower_bound or upper_bound
    
    return tuple(neighbour_idx)


def initial_population(N: int, p_S: int, N_I: int) -> np.ndarray: 
    # TODO: Cultists start in an area close together. 
    
    population = non_S_val * np.ones((N, N))
    population_possible_indices = np.arange(N)  # Same in x and y
    
    # Indices for S
    N_S = int(p_S * N)
    idx_S_x = rng.choice(a=population_possible_indices, size=N_S, replace=False)
    idx_S_y = rng.choice(a=population_possible_indices, size=N_S, replace=False)
    
    # Indices for I is chosen amongst 
    idx_chosen_by_S_x = np.in1d(population_possible_indices, idx_S_x)
    idx_chosen_by_S_y = np.in1d(population_possible_indices, idx_S_y)
    I_possible_indices_x = population_possible_indices[~idx_chosen_by_S_x]
    I_possible_indices_y = population_possible_indices[~idx_chosen_by_S_y]
    idx_I_x = rng.choice(a=I_possible_indices_x, size=N_I, replace=False)
    idx_I_y = rng.choice(a=I_possible_indices_y, size=N_I, replace=False)
    
    # Update values
    population[idx_S_x, idx_S_y] = S_val
    population[idx_I_x, idx_I_y] = I_val
    
    return population


def update_who_is_susceptible2d(population, p_S: float, p_non_S: float):
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
    S_idx = np.nonzero(population==S_val)  # Idx of susceptible people
    S_idx_size = S_idx[0].size  # Number of susceptible
    rng1 = rng.uniform(low=0, high=1, size=S_idx_size)  # Random number for each pair of indices
    S_change = rng1 < p_non_S  # Check which pairs change to non-S
    S_change_population_idx_x = S_idx[0][S_change]  # Get x population indices of those who change to non_S
    S_change_population_idx_y = S_idx[1][S_change]  # Get y population indices of those who change to non_S
    population[(S_change_population_idx_x, S_change_population_idx_y)] = non_S_val  # Update population
    
    # non-S -> S
    non_S_idx = np.nonzero(population==non_S_val)  # Get indicides of the non-Susceptible people
    non_S_idx_size = non_S_idx[0].size
    rng2 = rng.uniform(low=0, high=1, size=non_S_idx_size)  # Draw random number for each
    non_S_change = rng2 < p_S  # Check who changes to S i.e. undergoes crisis
    non_S_change_population_idx_x = non_S_idx[0][non_S_change]  # Get their indicis in the population array
    non_S_change_population_idx_y = non_S_idx[1][non_S_change]  # Get their indicis in the population array
    population[(non_S_change_population_idx_x, non_S_change_population_idx_y)] = S_val  # Update the population array
    
    return population
    

def evolve(N, p_S, p_non_S, N_I, time_steps):
    population = initial_population(N, p_S, N_I)
    population_history = np.empty((N, N, time_steps))
    population_history[:, :, 0] = 1 * population
    
    for i in tqdm(range(1, time_steps)):
        # Pick random person. If cultists, pick random neighbour. If S, S->. If E, E->S
        idx_person_picked = rng.integers(low=0, high=N, size=2)
        neighbour_idx = get_neighbour_idx(idx_person_picked, N)
        idx_person_picked = tuple(idx_person_picked)
        person_picked_state = population[idx_person_picked]
        neighbour_state = population[neighbour_idx]
        
        # Perform conversion attempt
        neighbour_new_state = conversion(person_picked_state, neighbour_state)
        population[neighbour_idx] = neighbour_new_state
        
        # S and non-S changes back and forth
        population = update_who_is_susceptible2d(population, p_S, p_non_S)        
        
        # Store in history
        population_history[:, :, i] = 1 * population
    
    np.save(dir_path_output + "population_history.npy", arr=population_history)


def plot_imshow():
    # Load data
    population_history = np.load(dir_path_output+"population_history.npy")
    time_steps = population_history[0, 0, :].size
    pop_initial = population_history[:, :, 0]
    pop_final = population_history[:, :, -1]
    
    # Create custom colormap
    cmap = ListedColormap(["black", "grey", "orange", "red"])  # Non-S, S, E, I
    
    # Plot
    fig, (ax0, ax1) = plt.subplots(ncols=2)
    
    # Initial 
    ax0.imshow(pop_initial, cmap=cmap, vmin=non_S_val, vmax=I_val)
    ax0.set(title="Initial")
    # Final
    im = ax1.imshow(pop_final, cmap=cmap, vmin=non_S_val, vmax=I_val)
    ax1.set(title="Final")
    
    # Colorbar
    cbar = fig.colorbar(im)
    cbar.set_ticks([non_S_val, S_val, E_val, I_val])
    cbar.set_ticklabels(["Non-S", "S", "E", "I"])
    # Figtitle
    fig.suptitle(f"Time = {time_steps}")
    
    figname = dir_path_image + "image_plot.png"
    plt.savefig(figname)
    plt.show()
    
    
if __name__ == "__main__":
   # Parameters
    N_tot = 100
    N_I = N_tot // 4
    p_S = 0.25  # P(non-S -> S)
    p_non_S = 0.01  # P(S -> non_S)
    
    time_steps = 50_000
    
    # Run evolution
    get_data = True
    if get_data:
        evolve(N_tot, p_S, p_non_S, N_I, time_steps)
    
    # Visualize
    plot_imshow()