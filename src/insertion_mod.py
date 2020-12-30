"""Module to handle the proposed inserted curves to be descended.

The module exhibits ``global`` variables that are used to remember the
state of the insertion step.

Global variables
----------------
**known_curves** : list[:py:class:`src.classes.curve`]
    List of member curves from the current iterate of the DGCG algorithm that
    have not yet been descended.
**crossover_memory** : :py:class:`src.insertion_mod.ordered_list_of_lists`
    Object that keeps track of the crossover information between
    the found stationary curves.
**cycling_iter** : iterator
    Cycling iterator that keeps track on the number of consecutive crossover
    curves that have been proposed.
"""
# Standard imports
import itertools as it
import copy
import sys
import numpy as np

# Local imports
from . import classes, config

# Useful class
class ordered_list_of_lists:
    """Class to organize the found stationary curves and executed crossovers.

    Initializes with no arguments into an empty list of lists.

    Attributes
    ----------
    data : list[list[tuple[list[int], int]]]
        A list of size ``M``, the number of known stationary curves, which at
        the entry ``i`` contains a list of size ``M-i-1``. For ``i<j``, The
        entry ``[i][j-i-1]`` contains crossover information between the i-th
        and the j-th stationary curves. This information is a tuple with a list
        of integers representing the indexes of the proposed crossovers and an
        integer indicating the total number of crossovers.
    """
    def __init__(self):
        self.data = []

    def add_empty_element_in_index(self, i):
        """ Insert an empty list in the target location.

        Parameters
        ----------
        i : int
            index to insert an empty list.

        Notes
        -----
        The main effort is to shift all the known relationships when inserting
        a list in between.
        """
        num_after_elements = len(self.data) - i
        empty_list_of_lists = [[] for j in range(num_after_elements)]
        self.data.insert(i, empty_list_of_lists)
        # Update all the past lists
        for j in range(i):
            self.data[j].insert(i-j-1, [])

    def GET(self, i, j):
        """Get the crossover information between the stationary stationary
        curves.

        Parameters
        ----------
        i,j : int
            Indices of stationary curves. i < j.

        Returns
        -------
        tuple[list[int], int]
            The crossover information between the chosen curves.
        """
        return self.data[i][j-i-1]

    def POST(self, i, j, val):
        """Modify the crossover information between two stationary curves.

        Parameters
        ----------
        i,j : int
            Indices of stationary curves. i < j.
        val : tuple[list[int], int]

        Returns
        -------
        None
        """
        self.data[i][j-i-1] = val

def update_crossover_memory(index):
    """Updates the crossover memory by including a new stationary curve.

    This method is meant to be called outside this module, to modify the
    ``global`` variable crossover_memory, which is an instance of
    :py:class:`src.insertion_mod.ordered_list_of_lists`.

    Parameters
    ----------
    index : int
        Location to insert a new stationary curve on the known set.

    Returns
    -------
    None
    """
    global crossover_memory
    # Create new entry for this element
    crossover_memory.add_empty_element_in_index(index)


# global variables
known_curves = []
crossover_memory = ordered_list_of_lists()
cycling_iter = it.cycle(range(config.crossover_consecutive_inserts))


def initialize(current_measure):
    """Initializes the global variables at the beggining of each insertion step.

    Parameters
    ----------
    current_measure : :py:class:`src.classes.measure`
        The current iterate of the DGCG algorithm.

    Returns
    -------
    None
    """
    global known_curves
    global crossover_memory
    global cycling_iter
    known_curves = copy.deepcopy(current_measure.curves)
    crossover_memory = ordered_list_of_lists()
    cycling_iter = it.cycle(range(config.crossover_consecutive_inserts))


def propose(w_t, stationary_curves, energy_curves):
    """Propose a curve to be descended.

    There are three types of proposed curves to insert:
        1. The already known curves from the current solution.
        2. Random curves placed by selecting random times and random locations.
        3. Crossover curves, these are obtained by merging two good descended.
        candidates.

    Parameters
    ----------
    w_t : :py:class:`src.classes.dual_variable`
        Dual variable associated to the current iterate
    stationary_curves : list[:py:class:`src.classes.curve`]
        List of found stationary curves
    energy_curves : numpy.ndarray
        1-dimensional list of ordered floats with the respective
        Benamou-Brenier energy of ``stationary_curves``. See also
        :py:meth:`src.classes.curve.energy`.

    Returns
    -------
    :py:class:`src.classes.curve`
        A curve to be descended by the multistart descent method.

    Notes
    -----
    This method will first propose all the ``known_curves`` from the
    ``current_measure``. Then it will switch between proposing ``M``
    consecutive crossover curves if possible and then a random curve.  The
    parameter ``M`` is modulated by ``config.crossover_consecutive_inserts``.
    For the random insertion, see
    :py:meth:`src.insertion_mod.random_insertion`, For crossovers, see
    :py:meth:`src.insertion_mod.find_crossover`
    """
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
        if next(cycling_iter) != config.crossover_consecutive_inserts - 1:
            # Attempt to find crossover
            crossover_curve = find_crossover(stationary_curves, energy_curves,
                                             w_t)
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
    w_t : :py:class:`src.classes.dual_variable`
        The dual variable associated to the current iterate of the algorithm.

    Returns
    -------
    :py:class:`src.classes.curve`
        A random curve.

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
        if w_t.get_sum_maxs()*config.insertion_length_bound_factor < rand_curve.energy():
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
    """ Rejection sampling over a density defined by the dual variable.

    Parameters
    ----------
    t : int
        Index of time sample. Takes values between 0,1,...,T-1
    w_t : :py:class:`src.classes.dual_variable`
        Dual variable associated with the current iterate.

    Returns
    -------
    numpy.ndarray
        A random point in Ω = [0,1]^2.
    """
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

def switch_at(curve1, curve2, idx):
    """Generate two crossover curves by switching at given time sample

    Parameters
    ----------
    curve1, curve2 : :py:class:`src.classes.curve`
        Curve to crossover
    idx : int
        Time sample index where the crossover happens.

    Returns
    -------
    new_curve_1 : :py:class:`src.classes.curve`
    new_curve_2 : :py:class:`src.classes.curve`
    """
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
    """Obtain all the crossovers between two curves.

    Parameters
    ----------
    curve1, curve2 : :py:class:`src.classes.curve`
        Curve to crossover.

    Returns
    -------
    list[:py:class:`src.classes.curve`]

    Notes
    -----
    To obtain a crossover, a minimum distance threshold is set by
    ``config.crossover_max_distance``. Then for every time these curves
    get closer than this and then separate, two new crossover curves are
    obtained.
    """
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

def find_crossover(stationary_curves, energy_curves, w_t):
    """Finds a crossover curve to propose from the list of stationary curves.

    Parameters
    ----------
    stationary_curves : list[:py:class:`src.classes.curve`]
        List of found stationary curves.
    energy_curves : numpy.array
        1-dimensional array of respective energies of the stationary curves.
    w_t : :py:class:`src.classes.dual_variable`.
        Dual variable associated to the current iterate.

    Returns
    -------
    :py:class:`src.classes.curve` or None
        If a crossover is found, returns it. If not, returns None.
    """
    # From the known tabu curves, and the crossover_table, attempt to find
    # a new crossover curve
    global crossover_memory
    def F(curve):
        # Define the energy here to evaluate the crossover children
        return -curve.integrate_against(w_t)/curve.energy()
    #
    crossover_search_attempts = config.crossover_search_attempts
    N = len(stationary_curves)
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
            children = crossover(stationary_curves[i], stationary_curves[j])
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
                crossover_list = crossover(stationary_curves[i],
                                           stationary_curves[j])
                return crossover_list[selection]
        attempts = attempts + 1
    # Failed to find unproposed crossover curves
    print("Couldn't find crossover curves to propose")
    return None

