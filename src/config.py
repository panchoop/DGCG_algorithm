""" General configuration file.

Summary
-------
This module contains all the configuration parameters that define the details
of the DGCG algorithm. Al parameters are set at execution of
:py:meth:`src.DGCG.solve` and then remain fixed.

Therefore, to modify any of these parameters, do it before executing
:py:meth:`src.DGCG.solve`
    Switch to use the ffmpeg library. This is required to save the obtained
    curves and measures as videos.
"""
# Standard imports
import pickle
import numpy as np


def self_pickle(filename):
    """ Function to pickle and save the variables in this module.

    The pickled variables are saved into the folder set by
    :py:data:`src.config.results_folder`
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
"""
str. By default, the algorithm stores at each iteration the iterate, graphs
the convergence plots, dual gaps, found stationary points, etc.
This variable indicates the name of the folder in which these are stored.
"""
logger = None
"""
:py:class:`src.log_mod.logger`.
The logger class is involved in all the logging activities, like plotting,
pickling data, terminal printing, etc. A logger object is created and then
accessed by all the modules here.
"""

# Time discretization values
T = 51
"""
int.
The number of time samples of the problem.
"""
time = np.linspace(0, 1, T)
"""
numpy.ndarray.
    The respective time samples of the problem.
"""
time_weights = np.ones(T)/T
"""
numpy.ndarray.
    The associated weights to each time sample. By default, these are equally
    weighted summing up to 1. Relevant when dealing with different uncertainty
    values for each time sample.
"""

# Problem coefficients
alpha = 0.1
"""
float.
Regularization coefficient of the problem
"""
beta = 0.1
"""
float.
Regularization coefficient of the problem
"""
# Problem data
f_t = None
"""
list[numpy.ndarray].
Input data in the problem, represents a list of elements in H_t for each t.
"""

# Measures parameters
measure_coefficient_too_low = 1e-18
"""
float.
The measure class is a weighted sum of atoms. When the weight of an
atom is lower than this threshold, it is automatically discarded.
"""

# Whole algorithm parameters
full_max_iterations = 1000
"""
int.
Maximum number of iterations of the algorithm.
"""

# Random insertion of curves parameters
insertion_max_segments = 20
"""
int.
In the insertion step, during the multistart gradient descent, random
curves are proposed for descense in insertion_mod.random_insertion
The number of segments of the random curves is chosen at random, with
this parameter defining the upper limit on the chosen segments.
"""
rejection_sampling_epsilon = 0.05
"""
float.
When generating random curves for insertion at
insertion_mod.random_insertion, once the time nodes of the curve is
defined, the spatial positions are chosed via the rejection_sampling
algorithm. This parameter is involved in the definition of used function.
In principle, the higher is this number, the faster the rejection sampling
algorithm will find a candidate. But simultaneously, it will miss possible
candidates that have values barely above 0.
"""
insertion_length_bound_factor = 1.1
"""
float.
When proposing curves to descend in insertion_mod.propose, it is known
from the theory that any solution must not exceed a certain length that
can be computed. If any proposed curve surpases this limit by a factor
given by this parameter, it is automatically discarded.
"""
multistart_pooling_num = 1000
"""
int.
When proposing random curves, many random curves are proposed and
afterwards, before descending them, we choose the best one from this group
The size of the generated random curves is defined by this parameter.
The criteria to choose the best curve is one that has the least F(γ) value.
"""

# Crossover parameter
crossover_consecutive_inserts = 30
"""
int.
The proposing method at insertion_mod.propose switches between choosing
a crossover curve or a random curve. For each N crossover propositions
it does 1 random proposition. N here corresponds to this parameter.
"""
crossover_search_attempts = 1000
"""
int.
To crossover curves the algorithm must look for curves that are close
enough to crossover and then check if these have been crossover beforehand.
This information is contained in the sort-of-dictionary object
insertion_mod.ordered_list_of_lists, and to look for new pairs it will
randomly access the entries to see if a crossover can be obtained.
It will attempt this random entries the number given by the his parameters,
if no crossover is found after this search, insertion_mod.propose will
declare that there are no available crossovers and then will propose a
random curve for descent.
"""
crossover_child_F_threshold = 0.8
"""
float.
Obtained crossover curves will be proposed for descensen only if their
energy F(γ) is close to the best known stationary curve. How close it has
to be is modulated by this parameter, it must satisfy
F(crossover_child) < crossover_child_F_threshold * F(best_curve),
remember that the energies are negative.
"""
crossover_max_distance = 0.05
"""
float.
Childs from two curves can be obtained only if at some point in time they
get close one to another, this parameter indicates how close they need to
get in H^1 norm for a crossover to happen.
"""

# Insertions step
insertion_eps = 1e-10
"""
float.
This is the tolenrance value to stop the algorithm. If the dual gap drops
below it, the algorithm exits.
"""

# multistart search iteration parameters
insertion_max_restarts = 20
"""
int.
The maximum number of restarts of the multistart algorithm.
"""
insertion_min_restarts = 15
"""
int.
The minimum number of restarts of the multistart algorithm. This
parameter is useful only in the case an early stop criteria is set
via the `multistart_early_stop` parameter.
"""
multistart_inter_iteration_checkup = 50
"""
int.
While descending a single curve during the multistart gradient descent,
the code will routinely check if curve being descended is close to the any
element of the stationary point set. If so, the descense is stopped
and the curve is discarded. This parameter regulates how often this
check is done. Precaution: The algorithm also is coded to "omit" the curves
that got too fast too close to the stationary point set. By "omiting", we
mean that such a descented curve will not count towards the number of
descented curves; "too fast" means that the curve got too close to the
statonary set before the first checkup. A consequence of this is that if
this checkup number is set too high, and there are a few stationary points,
then  (almost) all the descended curves will converge faster than the first
checkup and as such, they will not count towards the number of attempted
tries. Heavily slowing down the algorithm.
"""
multistart_max_discarded_tries = 30
"""
int.
If more than multistart_max_discarded_tries curves are discarded
consecutively. Then the algorithm will issue a warning to set
`multistart_inter_iteration_checkup` higher and will add a counter
to the number of restarts. This is a failsafe against a `while true` loop.
"""
multistart_taboo_dist = 0.01
"""
float.
The distance, in H^1 norm, of a curve to an element of the stationary
set to be discarded.
"""
multistart_energy_dist = 0.01
"""
float.
Acceleration parameter to measure the distance between the descended curve
with those of the stationary set. The stationary point set is ordered by
their F(γ) value, which is also readily available in a list. Therefore by
computing the F(γ) value of the descended curve, one can just compare the
current curve with those around that value, this parameter defines that
radius.
"""
multistart_early_stop = lambda num_tries, num_found: np.inf
"""
callable[[int,int], int], default constant equal to infinite.
This parameter allows to pass an early stop criteria to the multistart
algorithm. The input is a two variable function whose first input is
the number of attempted restarts, and the second parameter is the number
of found stationary point. The multistart gradient descent will stop once
it either reaches the `insertion_max_restart` value, or the value given by
this function.
"""
multistart_proposition_max_iter = 10000
"""
int.
Each proposed curve must start with negative energy, if it does not, it
is discarded and another curve is proposed. This parameter sets a limit on
how many attempts will be done.
"""

# multistart gradient descent parameters
multistart_descent_max_iter = 16000
"""
int.
This parameter limits the number of gradient descent steps that will be
done on each descended curve.
"""
multistart_descent_soft_max_iter = 5000
"""
int.
This is a soft maximum number of iterations. If the currently descended
curve has done more than this number of iterations, and simultaneously its
energy is not "good enough", then the descense will be stopped.
"""
multistart_descent_soft_max_threshold = 0.8
"""
float.
Sets the threshold to discard the current descended curve, the current
descended curve has to be at least this ratio closer to the best known
stationary curve.
"""
multistart_descent_init_step = 1e1
"""
float.
The gradient descent uses an Armijo with backtracking descent. This
parameter sets the intiial stepsize/
"""
multistart_descent_limit_stepsize = 1e-20
"""
float.
The gradient descent stops when the stepsize becomes smaller than this
value.
"""

# Quadratic optimization step
H1_tolerance = 1e-5
"""
float.
The quadratic optimization step will attempt to merge curves that are
closer than this distance in H1 norm.
"""
curves_list_length_lim = 1000
"""
int.
The quadratic optimization step will take at most this number of stationary
point found in the insertion step.
"""
curves_list_length_min = 10
"""
int.
In the optimization step after the insertion step, the inserted curves are
the union of the already known curves, together with those found in the
multistart descent. This parameter sets least number of stationary curves
from the mutlistart descent that have to be added for optimization.
"""
CVXOPT_TOL = 1e-25
"""
float.
CVXOPT is the used solver to tackle the quadratic optimization step. This
parameter defines the considered tolerance value for both the relative and
absolute errors.
"""

# Gradient flow + coefficient optimization parameters
slide_opt_max_iter = 100000
"""
int.
During the sliding step, this parameter modules the maximum number of
iterations to execute.
"""
slide_opt_in_between_iters = 100
"""
int.
During the sliding step, in between iterations, the weights of the measure
are optomized via the optimization step. This parameter regulates how often
this is done.
"""
slide_init_step = 1e0
"""
float.
The initial stepsize of the Armijo with Backtracking gradient descent
for the Sliding step.
"""
slide_limit_stepsize = 1e-20
"""
float, default 1e-20.
During the sliding step, the descent stops once the stepsize reaches this
size.
"""

# Logging parameters
log_output = False
"""
bool.
Switch to log the convergence information into a .txt file into the
`results` folder. WARNING: requires rework, too many useless lines are
saved.
"""
save_output_each_N = 1000
"""
int.
How often the saved logs will be saved. This parameter consider the number
of lines of the file.
"""
log_maximal_line_size = 10000
"""
int.
Maximum size of the logfile. If exceeded, the file is discarded.
"""

# Miscelaneous
use_ffmpeg = True
"""
bool.
Switch to enable/disable the use of ffmpeg. If disabled, no videos can be
saved.
"""
