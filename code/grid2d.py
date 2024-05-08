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
from tqdm import tqdm
from master import update_who_is_susceptible, rng, non_S_val, S_val, E_val, I_val, dir_path

# -- Parameters --
name = "grid_oscillation_S"
dir_path_image = dir_path + "image/" + name
dir_path_output = dir_path + "output/" + name


def attempt_conversion2d():
    
    return


def initial_population(N: int, p_S: int, N_I: int) -> np.ndarray: 
    # TODO: Cultists start in an area close together. 
    
    population = non_S_val * np.ones(N, N)
    population_possible_indices = np.arange(N)  # Same in x and y
    
    # Indices for S
    N_S = int(p_S * N ** 2)
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
    
    
def evolve(N, p_S, p_non_S, N_I, time_steps):
    population = initial_population(N, p_S, N_I)
    population_history = np.empty((N, N, time_steps))
    population_history[:, :, 0] = 1 * population
    
    for i in tqdm(range(1, time_steps)):
        # Pick random person. If cultists, pick random neighbour. If S, S->. If E, E->S
        idx_person_picked = rng.integers(low=0, high=N, size=2)
        neighbour_idx_relative_options = [(-1, 0), (0, -1), (1, 0), (0, 1)]
        neighbour_idx_relative = rng.choice(a=neighbour_idx_relative_options)
        neighbour_idx = idx_person_picked + neighbour_idx_relative
        
        # Perform conversion attempt
        attempt_conversion()
        
        # Oscillate S and non-S
        population = update_who_is_susceptible(population, p_S, p_non_S)        
        
        # Store in history
        population_history[:, :, i] = 1 * population

        
        
    
    np.savetxt(fname=dir_path_output + "population_history.gz", X=population_history)

