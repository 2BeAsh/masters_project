import numpy as np

# -- Parameters --
rng = np.random.default_rng()
non_S_val = -1
S_val = 0
E_val = 1
I_val = 2
dir_path = "code/"


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