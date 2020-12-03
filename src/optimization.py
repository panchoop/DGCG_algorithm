# Standard imports
import numpy as np
import copy
import code

# Not so standard imports
import cvxopt

# Local imports
from . import curves, config, insertion_mod
from . import operators as op

# Solver parameters
cvxopt.solvers.options['reltol']=1e-20
cvxopt.solvers.options['abstol']=1e-20
cvxopt.solvers.options['show_progress'] = False # to silence solver

def F(curve,w_t):
    # The evaluation of the operator F(γ) = W(γ)/L(γ)
    assert isinstance(curve, curves.curve) and isinstance(w_t, op.w_t)
    return -curve.integrate_against(w_t)/curve.energy()

def grad_F(curve, w_t):
    assert isinstance(curve, curves.curve) and isinstance(w_t, op.w_t)
    # We obtain the gradient of the operator F(γ) = W(γ)/L(γ), that is required
    # to minimize in the insertion step.
    # INPUTS: curve is a curve type object.
    #         w_t is a operators.w_t class object, i.e. a dual variable.
    L_gamma = curve.energy()
    W_gamma = -curve.integrate_against(w_t)
    # ∇L(γ) computation
    diff_positions = np.diff(curve.x, axis = 0) # γ_{i+1}-γ_{i} (T-1)x2 array
    diff_times = np.diff(config.time) #  t_{i+1}-t{i} 1D array
    diffs = np.diag(1/diff_times)@diff_positions # diff(γ)/diff(t) (T-1)x2 array
    prepend_zero_diffs = np.insert(diffs, 0,0, axis = 0)
    append_zero_diffs = np.insert(diffs, len(diffs), 0 , axis = 0)
    grad_L_gamma = config.beta*(prepend_zero_diffs - append_zero_diffs)
    # ∇W(γ) computation
    grad_W_gamma = -np.array([config.time_weights[t]*
                         w_t.grad_eval(t,curve.eval_discrete(t)).reshape(2)
                         for t in range(config.T)])
    # (L(γ)∇W(γ)-W(γ)∇L(γ))/L(γ)²
    pos_gradient = (L_gamma*grad_W_gamma - W_gamma*grad_L_gamma)/L_gamma**2
    gradient_curve = curves.curve(pos_gradient)
    return gradient_curve

def measure_trimming(current_measure, energy_curves = None):
    """ Trim a sparse measure by decreasing the number of members.

    This function takes an input measure and the proceeds to delete replicated
    atoms. If this is done in the insertion step, it furthers limits the number
    of members if they exceed a preset value (defined in config.py as
    curves_list_length_lim).
    There are two execution cases:
    case 1 - the vector energy_curves is provided, therefore it just checks for
    duplicates, deletes it and return.
    case 2 - the vector energy_curves is provided. In this case it is implicit
    that the trimming is ocurring during the insertion step, therefore the
    input measure is the union of both an original measure, with the stationary
    points. The stationary points are those that get further trummed if their
    energy (given by the energy_curves vector) is too low.

    :param current_measure: The target measure to be trimmed.
    :type current_measure: class:`src.curves.measure`
    :param energy_curves: list of F(γ) values for the curves obtained by the `multistart_descent` method, defaults to None.
    :type energy_curves: list, optional
    """
#    ---------------
#    Inputs:
#        current_measure (curves.measure type):
#            the measure to be trimmed.
#    Output:
#        curves_list (list of curves.curve type object):
#            The curves that survive the trimming. These are the atoms of the
#            new measure.
#    Kwargs:
#        energy_curves (list of np.float, default None):
#            If available, list of F(γ) values for the member curves of the
#            set of stationary points. It accelerates comparisons.
#    --------------
#    Preset parameters:
#        config.H1_tolerance
#        config.curves_list_length_lim
    curves_list = copy.deepcopy(current_measure.curves)
    if energy_curves == None:
        duplicates_idx = []
        for i, curve1 in enumerate(curves_list):
            for curve2 in curves_list[i+1:]:
                if (curve1-curve2).H1_norm() < config.H1_tolerance:
                    duplicates_idx.append(i)
                    break
        # take out the duplicated curves
        for i in reversed(duplicates_idx):
            curves_list.pop(i)
    else:
        # Get the number of current curves and tabu curves
        N_current_curves = len(curves_list)-len(energy_curves)
        N_stationary_curves = len(energy_curves)
        # separate curves
        current_curves = curves_list[0:N_current_curves]
        stationary_curves = curves_list[N_current_curves:]
        # Order the tabu curves
        sort_idx = np.argsort(energy_curves)
        stationary_curves = [stationary_curves[i] for i in sort_idx]
        energy_curves_list = [energy_curves[i] for i in sort_idx]
        # Eliminate duplicate curves, using the information of the energy_curves
        # (if possible) to accelerate this process
        duplicates_idx = []
        # The current curves should not be duplicated, we check with the 
        # tabu curves if they are duplicated.
        for curve1 in current_curves:
            for idx, curve2 in enumerate(stationary_curves):
                if (curve1-curve2).H1_norm() < config.H1_tolerance:
                    duplicates_idx.append(idx)
        ## eliminate duplicated idx's and sort them
        duplicates_idx = list(dict.fromkeys(duplicates_idx))
        duplicates_idx.sort(reverse=True)
        # remove the duplicate tabu curves
        for i in duplicates_idx:
            stationary_curves.pop(i)
            energy_curves_list.pop(i)
        print("Eliminating duplicas, eliminated {} duplicate tabu curves".format(
                                                        len(duplicates_idx)))
        # Tabu curves should not be replicated given the Tabu search algorith.
        # Now trim if the curves_list is too long
        pop_counter = 0
        while len(stationary_curves) + N_current_curves > config.curves_list_length_lim and \
              energy_curves_list[-1] >= -1:
            # find the one with least energy and pop it
            stationary_curves.pop()
            energy_curves_list.pop()
            pop_counter += 1
        if pop_counter > 0:
            print("Trimming process: {} low energy tabu curves eliminated".format(
                pop_counter))
        curves_list = current_curves + stationary_curves # joining two lists
    return curves_list

def solve_quadratic_program(current_measure, energy_curves = None):
    assert isinstance(current_measure, curves.measure)
    # Build the quadratic system of step 5 and then use some generic python
    # solver to get a solution.
    # Build matrix Q and vector b
    logger = config.logger
    # First, check that no curves are duplicated
    curves_list = measure_trimming(current_measure, energy_curves = energy_curves)
    N = len(curves_list)
    Q = np.zeros((N,N), dtype=float)
    b = np.zeros(N)
    for i,curve in enumerate(curves_list):
        measure_i = curves.measure()
        measure_i.add(curve, 1)
        K_t_i = op.K_t_star_full(measure_i)
        b[i] = op.int_time_H_t_product(K_t_i, config.f_t)
        for j in range(i,N):
            measure_j = curves.measure()
            measure_j.add(curves_list[j], 1)
            K_t_j = op.K_t_star_full(measure_j)
            entry = op.int_time_H_t_product(K_t_i, K_t_j)
            Q[i,j] = entry
            Q[j,i] = entry
    # Theoretically, Q is positive semi-definite. Numerically, it might not.
    # Here we force Q to be positive semi-definite for the cvxopt solver to work
    # this is done simply by replacing the negative eigenvalues with 0
    minEigVal = 0
    eigval, eigvec = np.linalg.eigh(Q)
    if min(eigval)<minEigVal:
        # truncate
        print("Negative eigenvalues: ",[eig for eig in eigval if eig < minEigVal])
        eigval = np.maximum(eigval, minEigVal)
        # Recompute Q = VΣV^(-1)
        Q2 = np.linalg.solve(eigvec.T, np.diag(eigval)@eigvec.T).T
        print("PSD projection relative norm difference:",
              np.linalg.norm(Q-Q2)/np.linalg.norm(Q))
        QQ = Q2
    else:
        QQ = Q
    try:
        Qc = cvxopt.matrix(QQ)
        bb = cvxopt.matrix(1 - b.reshape(-1,1))
        G  = cvxopt.matrix(-np.eye(N))
        h  = cvxopt.matrix(np.zeros((N,1)))
        sol = cvxopt.solvers.qp(Qc,bb,G,h)
        coefficients = np.array(sol['x']).reshape(-1)
    except Exception as e:
        print(e)
        print("Failed to use cvxopt, aborted.")
    # Incorporate as 0 coefficients those of the duplicates
    coefficients = list(coefficients)
    logger.status([1,2,2], coefficients)
    return curves_list, coefficients

def weight_optimization_step(current_measure, energy_curves = None):
    config.logger.status([1,2,1])
    # optimizes the coefficients for the current_measure
    # The energy_curves is a vector of energy useful for the trimming process
    assert isinstance(current_measure, curves.measure)
    curves_list, coefficients =\
            solve_quadratic_program(current_measure,
                                              energy_curves = energy_curves)
    new_current_measure = curves.measure()
    for curve, intensity in zip(curves_list, coefficients):
        new_current_measure.add(curve, intensity)
    return new_current_measure

def slide_and_optimize(current_measure):
    assert isinstance(current_measure, curves.measure)
    # Method that for a given measure, applies gradient flow on the current
    # curves to shift them, seeking to minimize the main problem's energy.
    # This method intercalates gradient flow methods and optimization steps.
    # Input and output: measure type object.
    stepsize = config.g_flow_init_step
    total_iterations = 0
    while total_iterations <= config.g_flow_opt_max_iter:
        current_measure, stepsize, iters = gradient_descent(current_measure,
                                                               stepsize)
        total_iterations += config.g_flow_opt_in_between_iters
        current_measure = weight_optimization_step(current_measure)
        if stepsize < config.g_flow_limit_stepsize:
            # The gradient flow converged, but since the coefficients got
            # optimized, it is required to restart the gradient flow.
            stepsize = np.sqrt(config.g_flow_limit_stepsize)
        if iters == 0:
            # The current measure is already optimal, therefore, there is no
            # need to keep iterating
            break
    return current_measure

def gradient_descent(current_measure, init_step,
                        max_iter = config.g_flow_opt_in_between_iters):
    # We apply the gradient flow to simultaneously perturb the position of all
    # the current curves defining the measure.
    # Input: current_measure, measure type object.
    #        max_iter > 0 integer number of iterations to perform
    # Output: measure type object and finishing stepsize.
    f_t = config.f_t
    logger = config.logger
    def full_gradient(current_measure):
        # Obtains the full gradient, which is an element in a H1 product space
        w_t = op.w_t(current_measure)
        curve_list = []
        for curve in current_measure.curves:
            curve_list.append( grad_F(curve,w_t))
        return curves.curve_product(curve_list, current_measure.intensities)
    # Stop when stepsize get smaller than
    limit_stepsize = config.g_flow_limit_stepsize
    def backtracking(current_measure, stepsize):
        current_curve_prod = current_measure.to_curve_product()
        decrease_parameter = 0.8
        control_parameter = 1e-15
        gradient = full_gradient(current_measure)
        m = control_parameter*gradient.H1_norm()
        while stepsize >= limit_stepsize:
            new_curve_list = current_curve_prod - stepsize*gradient
            new_measure = new_curve_list.to_measure()
            new_main_energy = new_measure.get_main_energy()
            current_energy  = current_measure.get_main_energy()
            if current_energy - new_main_energy > stepsize*m:
                break
            stepsize = stepsize*decrease_parameter
        return new_measure, stepsize
    # Descent implementation
    new_measure = current_measure
    current_energy = new_measure.get_main_energy()
    # the initial step considered for the algorithm
    stepsize = init_step
    for iters in range(max_iter):
        new_measure, stepsize = \
                backtracking(new_measure, stepsize*1.2)
        logger.status([3,0,0], new_measure, stepsize, iters)
        if stepsize < limit_stepsize:
            break
    logger.status([3,0,1])
    return new_measure, stepsize, iters

def dual_gap(current_measure, stationary_curves):
    """ Dual gap in the current measure.

    The dual gap computed using the Lemma formula for it. It recieves as
    input the current iterate of the algorithm, together with a list of
    curves ordered output of the taboo search. This list of curves is ordered
    by the energy F(γ), therefore the first member is the obtained minimizer
    of the insertion step problem.
    --------------------------
    Arguments:
        current_measure (measure class):
            The current iterate of the algorithm.
        stationary_curves (list of curves class):
            A list of curves output of the taboo search. It is assumed that
            the curves are ordered by increasing F(γ) values.

    Output:
        dual_gap (float):
            The dual gap computed in the current_measure folowing formula (??)

    Keyword arguments: None
    --------------------------
    """
    # Compute the dual variable
    w_t = op.w_t(current_measure)
    # Compute the constant (??)
    M_0 = op.int_time_H_t_product(config.f_t, config.f_t)/2
    # Extract the global minimizer
    insertion_step_minimizer = stationary_curves[0]
    # Build a measure with the global minimizer
    compare_measure = curves.measure()
    compare_measure.add(insertion_step_minimizer, 1)
    # Formula
    return M_0*(compare_measure.integrate_against(w_t)**2 - 1)/2


