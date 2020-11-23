# Standard imports
import numpy as np

# Parameter file

# Organizing parameters, temporal folder name
results_folder = 'results'

# Time discretization values
T = 51
time = np.linspace(0,1,T)
time_weights = np.ones(T)/T

# Problem coefficients
alpha = 1
beta  = 1
# Problem data
f_t = None

# Measures parameters
measure_coefficient_too_low = 1e-10

# Whole algorithm parameters
full_max_iterations = 1000

# Max_curve parameters
max_curve_x_res   = 1/100
max_curve_max_val_threshold = 0.90

# Random insertion of curves parameters
insertion_max_segments = 5
rejection_sampling_epsilon = 0.05
insertion_length_bound_factor = 1.1
multistart_pooling_num = 100

# Crossover parameter
crossover_consecutive_inserts = 30
crossover_search_attempts = 1000
crossover_child_F_threshold = 0.8
switching_max_distance = 0.05


# Step3 tabu search iteration parameters
insertion_max_restarts = 2
insertion_min_restarts = 15
step3_max_attempts_to_find_better_curve = 16 #warning, min < max
step3_max_number_of_failures = 50
step3_tabu_in_between_iteration_condition_checkup = 50
step3_tabu_dist = 0.1
step3_energy_dist = 0.01
multistart_early_stop = lambda n: np.log(0.01)/np.log((n-1)/n)


# Step3 gradient descent parameters
step3_descent_max_iter = 16000
step3_descent_soft_max_iter = 5000
step3_descent_init_step = 1e1
step3_descent_limit_stepsize = 1e-20

# Quadratic optimization step
H1_tolerance = 1e-5
H1_max_tolerance = 1e-2
H1_tol_increment = 10
energy_change_tolerance = 1e-16
curves_list_length_lim = 100

# Gradient flow + coefficient optimization parameters
g_flow_opt_max_iter = 100000
g_flow_opt_in_between_iters = 100
g_flow_init_step = 1e0
g_flow_limit_stepsize = 1e-20

# Logging parameters
log_output = False
save_output_each_N = 10000

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
on the Tabu set. (a low value implies a lot of wasted resources checking against
all the curves in the Tabu set, a high value implies wasting too much resources
descending a curve that clearly is converging to one in the tabu set).

"""
