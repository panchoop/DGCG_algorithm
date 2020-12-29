""" This module handles the proposed inserted curves to be descended. In short,
there are three types of proposed curves to insert:
    1) The already known curves from the current solution
    2) Random curves placed by selecting random times and random locations.
    3) Crossover curves, these are obtained by merging two good descended
    candidates.
"""
# Standard imports
import itertools as it
import copy
import sys
import numpy as np

# Local imports
from . import classes, config


# Settings
# Number of attempts of the mergestack before trying a new one
crossover_consecutive_inserts = config.crossover_consecutive_inserts
cycling_iter = it.cycle(range(crossover_consecutive_inserts))

# Useful class
class ordered_list_of_lists:
    # Ordered list of ordered lists that remember where the crossovers have
    # have ocurred and updates it's size when increasing the curve set by
    # appending in a sorted manner.
    def __init__(self):
        self.data = []

    def add_empty_element_in_index(self, i):
        # Insert an empty list of lists in the desired location
        num_after_elements = len(self.data) - i
        empty_list_of_lists = [[] for j in range(num_after_elements)]
        self.data.insert(i, empty_list_of_lists)
        # Update all the past lists
        for j in range(i):
            self.data[j].insert(i-j-1, [])

    def GET(self, i, j):
        # Find the information hold for the pair i,j, with i < j
        return self.data[i][j-i-1]

    def POST(self, i, j, val):
        # Insert information in target location
        self.data[i][j-i-1] = val

def update_crossover_memory(index):
    # Make an entry to an inserted curve in the i-th position
    global crossover_memory
    # Create new entry for this element
    crossover_memory.add_empty_element_in_index(index)


# global variables
known_curves = []
crossover_memory = ordered_list_of_lists()

# Main output function
def initialize(current_measure):
    # To be used at the beginning of the insertion step.
    global known_curves
    global crossover_memory
    known_curves = copy.deepcopy(current_measure.curves)
    crossover_memory = ordered_list_of_lists()

def propose(w_t, tabu_curves, energy_curves):
    global known_curves
    global cycling_iter
    global crossover_memory
    # 1) Descend the known curves, these are those of the current solution
    if len(known_curves) > 0:
        print("Proposing known curve")
        return known_curves.pop()
    # 2) If there is someone interesting on the merge stack, descended that one
    else:
        # See if it is crossover turn
        if next(cycling_iter) != crossover_consecutive_inserts - 1:
            # Attempt to find crossover
            crossover_curve = find_crossover(tabu_curves, energy_curves, w_t)
            if crossover_curve is not None:
                # If crossover is found, propose it
                print("Proposing crossover curve")
                return crossover_curve
            else:
                # If none is found, propose random curve
                return random_insertion(w_t)
        else:
            # 3) Else propose a random curve
            return random_insertion(w_t)


def random_insertion(w_t):
    """ Method that proposes a random curve to be descended.

    It selects a random number of time samples (controled via
    config.insertion_max_segments) and then to select the spatial points
    of the proposed curve, it uses the rejection-sampling algorithm using
    as information the input dual variable w_t.

    Parameters
    ----------
    w_t : operators.w_t class object
        Represents the dual variable of the current state of the algorithm.

    Returns
    -------
    classes.curve class object, a random curve.

    Notes
    -----
    For further information, check the paper that defined this code.
    """
    logger = config.logger
    min_segments = 1
    max_segments = min(config.T-1, config.insertion_max_segments)

    def sample_random_curve(w_t):
        num_segments = np.random.randint(max_segments-min_segments+1)\
                                        + min_segments
        # preset the intermediate random times
        if num_segments > config.T + 1:
            sys.exit('More segments than available times. ' +
                     'Decrease config.insertion_max_segments')
        considered_times = [0, config.T-1]
        while len(considered_times) <= num_segments:
            new_time = np.random.randint(config.T)
            if not (new_time in considered_times):
                considered_times.append(new_time)
        considered_times = np.sort(np.array(considered_times), -1)
        # times
        positions = rejection_sampling(0, w_t)
        for t in considered_times[1:]:
            positions = np.append(positions, rejection_sampling(t, w_t), 0)
        rand_curve = classes.curve(considered_times/(config.T - 1), positions)
        # discarding any proposed curve that has too much length
        if w_t.sum_maxs*config.insertion_length_bound_factor < rand_curve.energy():
            logger.status([1, 1, 1, 2], considered_times)
            return sample_random_curve(w_t)
        else:
            return rand_curve
    tentative_random_curves = []
    tentative_random_curves_energy = []
    pool_number = config.multistart_pooling_num
    def F(curve):
        # Define the energy here to evaluate the crossover children
        return -curve.integrate_against(w_t)/curve.energy()
    for i in range(pool_number):
        rand_curve = sample_random_curve(w_t)
        tentative_random_curves.append(rand_curve)
        tentative_random_curves_energy.append(F(rand_curve))
    # select the one with the best energy
    idx_best = np.argmin(tentative_random_curves_energy)
    return_curve = tentative_random_curves[idx_best]
    return_energy = tentative_random_curves_energy[idx_best]
    logger.status([1, 1, 1, 1], return_energy)
    # Record statistics of the produced curve
    return return_curve

def rejection_sampling(t, w_t):
    # Rejection sampling over a density defined by w_t
    # First, consider an epsilon below 0 such that we consider
    # that no endpoint of a curve would lie. 
    support, density_max = w_t.as_density_get_params(t)
    M = support*density_max
    iter_reasonable_threshold = 10000
    iter_index = 0
    while iter_index < iter_reasonable_threshold:
        # sample from uniform distribution on the support of w_t as density.
        reasonable_threshold = 10000
        i = 0
        while i < reasonable_threshold:
            x = np.random.rand()
            y = np.random.rand()
            sample = np.array([[x, y]])
            y = w_t.as_density_eval(t, sample)
            if y > 0:
                break
            else:
                i = i + 1
        if i == reasonable_threshold:
            sys.exit('It is not able to sample inside the support of w_t')
        # sample rejection sampling
        u = np.random.rand()
        if u < y/M*support:
            # accept
            return sample
        else:
            # reject
            iter_index = iter_index+1
    sys.exit(('The rejection_sampling algorithm failed to find sample in {} ' +
             'iterations').format(iter_index))

def curve_smoother(curve):
    assert isinstance(curve, classes.curve)
    # Method that for a given curve, it gives an smoother alternative. It is
    # achieved by averaging each point with the neighbours.
    # Input: curve type object.
    # Output: curve type object.
    points = curve.spatial_points
    new_points = points
    for i in range(1, len(points)-1):
        new_points[i] = (points[i-1]+points[i+1])/2
    return classes.curve(curve.time_samples, new_points)

def switch_at(curve1, curve2, idx):
    # Method that given a particular time index, produces the two curves
    # obtained by switching at that position
    midpoint = (curve1.spatial_points[idx] + curve2.spatial_points[idx])/2
    midpoint = midpoint.reshape(1, -1)
    tail_x1 = curve1.spatial_points[:idx, :]
    tail_x2 = curve2.spatial_points[:idx, :]
    head_x1 = curve1.spatial_points[idx+1:, :]
    head_x2 = curve2.spatial_points[idx+1:, :]
    new_x1 = np.vstack((tail_x1, midpoint, head_x2))
    new_x2 = np.vstack((tail_x2, midpoint, head_x1))
    new_curve1 = classes.curve(curve1.time_samples, new_x1)
    new_curve2 = classes.curve(curve2.time_samples, new_x2)
    return new_curve1, new_curve2

def crossover(curve1, curve2):
    # Method that given two curves, attempts to "mix them" by generating
    # other two curves, that correspond to start from one curve and transition
    # into the path of the other.
    diff_loc = curve1.spatial_points - curve2.spatial_points
    norms = np.linalg.norm(diff_loc, axis=1)
    # Then recognize the jumps: 1 if they were apart and got close
    #                           -1 if they were close and got far apart
    #                           0 if nothing happened
    jumps = np.diff((norms <= config.crossover_max_distance).astype(int))
    if len(jumps) == 0:
        # if there are no jumps, do not return
        return []
    # We want the indexes with 1s
    jump_idx = np.where(jumps == 1)[0] + 1
    # And we need to discard the last one if they stayed close until the end
    if norms[-1] <= config.crossover_max_distance:
        jump_idx = jump_idx[:-1]
    # We have the index locations for the switchings
    curve_descendants = []
    for idx in jump_idx:
        new_curve1, new_curve2 = switch_at(curve1, curve2, idx)
        curve_descendants.append(new_curve1)
        curve_descendants.append(new_curve2)
    return curve_descendants

def find_crossover(tabu_curves, energy_curves, w_t):
    # From the known tabu curves, and the crossover_table, attempt to find
    # a new crossover curve
    global crossover_memory
    def F(curve):
        # Define the energy here to evaluate the crossover children
        return -curve.integrate_against(w_t)/curve.energy()
    #
    crossover_search_attempts = config.crossover_search_attempts
    N = len(tabu_curves)
    attempts = 0
    while N > 1 and attempts < crossover_search_attempts:
        # i<j in for index search
        i = np.random.randint(N-1)
        j = np.random.randint(i+1, N)
        we_remember = crossover_memory.GET(i, j)
        # This is a tuple ( list, int), int indicates the number of children
        # list contains integers indicating the already proposed children.
        if we_remember == []:
            # These have never been crossover.
            # Crossover them and check the energies of the generated children
            children = crossover(tabu_curves[i], tabu_curves[j])
            if len(children) == 0:
                # empty children from these crossover
                crossover_memory.POST(i, j, ([], 0))
            else:
                # There are children! 
                # We know calculate the energy of the children and see if it
                # is acceptable or not.
                proposed = []
                for idx, child in enumerate(children):
                    child_threshold = config.crossover_child_F_threshold
                    if F(child) > child_threshold*energy_curves[0]:
                        # The child has not good enough energy, discarded
                        # (by setting it as an already proposed one)
                        proposed.append(idx)
                    else:
                        # The child has good enough energy!
                        pass
                we_remember = (proposed, len(children))
                if len(we_remember[0]) == we_remember[1]:
                    # All the children had unnacceptable energy. We pass
                    crossover_memory.POST(i, j, we_remember)
                else:
                    # There are children with acceptable energy!
                    unproposed = [i for i in range(we_remember[1])
                                  if i not in we_remember[0]]
                    # select a random one
                    selection = np.random.choice(unproposed)
                    # Edit dictionary to remember it was proposed
                    we_remember[0].append(selection)
                    crossover_memory.POST(i, j, we_remember)
                    # Crossover and return the requested children
                    return children[selection]
        # If these have already being crossover
        else:
            if len(we_remember[0]) == we_remember[1]:
                # Case in which we have already proposed all the crossovers
                # between (i,j). We search again
                pass
            else:
                # There are un-proposed children!
                unproposed = [i for i in range(we_remember[1])
                              if i not in we_remember[0]]
                # select a random one
                selection = np.random.choice(unproposed)
                # edit dictionary
                we_remember[0].append(selection)
                crossover_memory.POST(i, j, we_remember)
                # Crossover and return the requested children
                return crossover(tabu_curves[i], tabu_curves[j])[selection]
        attempts = attempts + 1
    # Failed to find unproposed crossover curves
    print("Couldn't find crossover curves to propose")
    return None

