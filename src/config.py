""" General configuration file.
Summary
-------
This module contains all the configuration parameters that define the details
of the DGCG algorithm. Al parameters are set at execution of `DGCG.solve`
and then remain fixed.

Members
-------
results_folder : str, default 'results'
    By default, the algorithm stores at each iteration the iterate, graphs
    the convergence plots, dual gaps, found stationary points, etc.
    This variable indicates the name of the folder in which these are stored.
logger : misc.logger class, default None
    The logger class is involved in all the logging activities, like plotting,
    pickling data, terminal printing, etc. A logger object is created and then
    accessed by all the modules here via `config.logger`.
T : int, default 51
    The number of time samples of the problem.
time : numpy.ndarray, default np.linspace(0, 1, T)
    The respective time samples of the problem.
time_weights : numpt.ndarrat, default np.ones(T)/T
    The associated weights to each time sample. By default, these are equally
    weighted summing up to 1. Relevant when dealing with different uncertainty
    values for each time sample.
alpha : numpy.float, default 0.1
    Regularization coefficient of the problem
beta : numpy.float, default 0.1
    Regularization coefficient of the problem
f_t : list of numpy.ndarray, default None
    Input data in the problem, represents a list of elements in H_t for each t.
measure_coefficient_too_low : numpy.float, default 1e-18
    The measure class is a weighted sum of atoms. When the weight of an
    atom is lower than this threshold, it is automatically discarded.
full_max_iteration : int, default 1000
    Maximum number of iterations of the algorithm.
insertion_max_segments : int, default 20
    In the insertion step, during the multistart gradient descent, random
    curves are proposed for descense in insertion_mod.random_insertion
    The number of segments of the random curves is chosen at random, with
    this parameter defining the upper limit on the chosen segments.
rejection_sampling_epsilon : numpy.float, default 0.05
    When generating random curves for insertion at
    insertion_mod.random_insertion, once the time nodes of the curve is
    defined, the spatial positions are chosed via the rejection_sampling
    algorithm. This parameter is involved in the definition of used function.
    In principle, the higher is this number, the faster the rejection sampling
    algorithm will find a candidate. But simultaneously, it will miss possible
    candidates that have values barely above 0.
insertion_length_bound_factor : numpy.float, default 1.1
    When proposing curves to descend in insertion_mod.propose, it is known
    from the theory that any solution must not exceed a certain length that
    can be computed. If any proposed curve surpases this limit by a factor
    given by this parameter, it is automatically discarded.
multistart_pooling_number : int, default 1000
    When proposing random curves, many random curves are proposed and
    afterwards, before descending them, we choose the best one from this group
    The size of the generated random curves is defined by this parameter.
    The criteria to choose the best curve is one that has the least F(γ) value.
crossover_consecutive_inserts : int, default 30
    The proposing method at insertion_mod.propose switches between choosing
    a crossover curve or a random curve. For each N crossover propositions
    it does 1 random proposition. N here corresponds to this parameter.
crossover_search_attempts : int, default 1000
    To crossover curves the algorithm must look for curves that are close
    enough to crossover and then check if these have been crossover beforehand.
    This information is contained in the sort-of-dictionary object
    insertion_mod.ordered_list_of_lists, and to look for new pairs it will
    randomly access the entries to see if a crossover can be obtained.
    It will attempt this random entries the number given by the his parameters,
    if no crossover is found after this search, insertion_mod.propose will
    declare that there are no available crossovers and then will propose a
    random curve for descent.
crossover_child_F_threshold : numpy.float, default 0.8
    


    

Extensive description of each parameter of this module

"""
# Standard imports
import pickle
import numpy as np


def self_pickle(filename):
    """ Function to pickle and save the variables in this module.

    In general, one could just look at this file to know the parameters. In
    practice, one will modify these values on the fly using the DGCG controler.
    Therefore it is better to have a method to read and save these settings
    automatically right before execution.
    """
    exclude_list = ['np',  'pickle', 'self_pickle', 'filename', 'logger',
                    'multistart_early_stop']
    variabls = [var for var in globals()
                if var[:2] != '__' and var not in exclude_list]
    var_dict = {}
    for var in variabls:
        var_dict[var] = globals()[var]
    pickling_on = open(filename, 'wb')
    pickle.dump(var_dict, pickling_on)


# Organizing parameters, temporal folder name
results_folder = 'results'
logger = None

# Time discretization values
T = 51
time = np.linspace(0, 1, T)
time_weights = np.ones(T)/T

# Problem coefficients
alpha = 0.1
beta = 0.1
# Problem data
f_t = None

# Measures parameters
measure_coefficient_too_low = 1e-18

# Whole algorithm parameters
full_max_iterations = 1000

# Random insertion of curves parameters
insertion_max_segments = 20
rejection_sampling_epsilon = 0.05
insertion_length_bound_factor = 1.1
multistart_pooling_num = 1000

# Crossover parameter
crossover_consecutive_inserts = 30
crossover_search_attempts = 1000
crossover_child_F_threshold = 0.8
switching_max_distance = 0.05

# Insertions step
insertion_eps = 1e-10

# multistart search iteration parameters
insertion_max_restarts = 20
insertion_min_restarts = 15
multistart_inter_iteration_checkup = 50
multistart_taboo_dist = 0.01
multistart_energy_dist = 0.01
multistart_early_stop = lambda num_tries, num_found: np.inf
multistart_proposition_max_iter = 10000
multistart_max_discarded_tries = 30

# multistart gradient descent parameters
multistart_descent_max_iter = 16000
multistart_descent_soft_max_iter = 5000
multistart_descent_soft_max_threshold = 0.8
multistart_descent_init_step = 1e1
multistart_descent_limit_stepsize = 1e-20

# Quadratic optimization step
H1_tolerance = 1e-5
curves_list_length_lim = 1000
curves_list_length_min = 10
CVXOPT_TOL = 1e-25

# Gradient flow + coefficient optimization parameters
g_flow_opt_max_iter = 100000
g_flow_opt_in_between_iters = 100
g_flow_init_step = 1e0
g_flow_limit_stepsize = 1e-20

# Logging parameters
log_output = False
save_output_each_N = 1000
log_maximal_line_size = 10000

""" PARAMETER EXPLANATION GUIDE:

* Problem coefficients
alpha, beta > 0, are the regularization parameters of the underlying problem.

* Curve and measures parameters
curves_times_samples: the considered time discretization for the considered
time-continuous curves in the time-continuous version of the problem
measure_coefficient_too_low > 0, if a coefficient associated to some of the
curves is too small, we consider the particular coefficient to be zero instead.

* Whole algorithm parameters
full_max_iteration. A complete iteration consists of an insertion step,
merging step and flowing step. This number limits the number of complete
iterations of the algorithm.

* Max_curve parameters
max_curve_x_res > 0 stands for the spatial resolution of the max_curve. The max
curve is a curve that passes for each time through the maximum of the function
w_t. Since afterwards the algorithm procedes to do a gradient descent, this
maximum values can be chosen in a "less precise" way, therefore, instead of
expensively finding the maximum at each step, a predefined spatial resolution
is chosen and then the function w_t is discreetly sampled on a spatial grid
with width defined by the max_curve_x_res parameter.

* Step3 tabu search iteration parameters
- step3_min_attempts_to_find_better_curve,
- step3_max_attempts_to_find_better_curve
At the step3, we need to find a curve that minimizes the target step3_energy.
The problem is smooth but not convex. Therefore, the proposed approach is to
shoot some curves and then descend them. By doing so, we are able to find local
minima of the target functional. Empirically, it seems that it is not required
to have the precise minimum of the functional, so these parameters basically
allow to accelerate the algorithm trade-offing some sloppyness.
step3_min_attempts_to_find_better_curve stands for the minimum number of
attempts taken by the algorithm to find an acceptable curve to insert.
If the algorithm does not find an acceptable curve to insert after this minimum
number of tries, it will keep trying to find better candidates until reaching
step3_max_attempts_to_find_better_curve. If this number is reached and no
acceptable curve was found, the algorithm considers the true minimum to be
already visited, and therefore the algorithm stops.
- step3_tabu_in_between_iteration_condition_checkup
- step3_tabu_dist
The tabu search has an optimization step in which no all curves are descended
to the fullest, as it is clear that they are descending to an already known
local minimum. To do it so, the H1 norm is evaluated from the current curve
candidate and those in the tabu set, the threshold in which to decide that the
curve will descent to any already known local minimum curve is step3_tabu_dist.
step3_tabu_in_between_iteration_condition_checkup is a parameter indicating
after how many iterations to check if the current curve is close to someone
on the Tabu set. (a low value implies a lot of wasted resources checking
against all the curves in the Tabu set, a high value implies wasting too much
resources descending a curve that clearly is converging to one in the tabu
set).

"""
