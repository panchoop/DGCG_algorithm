# Standard imports
import copy
import numpy as np

# Not so standard imports
import cvxopt

# Local imports
from . import curves, config
from . import operators as op

# Solver parameters
cvxopt.solvers.options['reltol'] = config.CVXOPT_TOL
cvxopt.solvers.options['abstol'] = config.CVXOPT_TOL
cvxopt.solvers.options['show_progress'] = False  # to silence solver


def F(curve, w_t):
    """ The F(γ) operator, defined as F(γ) = W(γ)/L(γ).

    Parameters
    ----------
    curve : DGCG.curves.curve class
    w_t : DGCG.operators.w_t class

    Returns
    -------
    double number.

    Notes
    -----
    When solving the insertion step, this is the main energy to minimize.
    """
    assert isinstance(curve, curves.curve) and isinstance(w_t, op.w_t)
    return -curve.integrate_against(w_t)/curve.energy()


def grad_F(curve, w_t):
    """ The gradient of the F operator, ∇F(γ).

    Parameters
    ----------
    curve : DGCG.curves.curve class
    w_t : DGCG.operators.w_t class

    Returns
    -------
    double number.

    Notes
    -----
    We use the gradient to minimize F(γ).
    """
    assert isinstance(curve, curves.curve) and isinstance(w_t, op.w_t)
    L_gamma = curve.energy()
    W_gamma = -curve.integrate_against(w_t)
    # ∇L(γ) computation
    diff_positions = np.diff(curve.x, axis=0)  # γ_{i+1}-γ_{i} (T-1)x2 array
    diff_times = np.diff(config.time)   # t_{i+1}-t{i} 1D array
    diffs = np.diag(1/diff_times)@diff_positions  # diff(γ)/diff(t) (T-1)x2 array
    prepend_zero_diffs = np.insert(diffs, 0, 0, axis=0)
    append_zero_diffs = np.insert(diffs, len(diffs), 0, axis=0)
    grad_L_gamma = config.beta*(prepend_zero_diffs - append_zero_diffs)
    # ∇W(γ) computation
    t_weigh = config.time_weights
    w_t_curve = lambda t: w_t.grad_eval(t, curve.eval_discrete(t)).reshape(2)
    T = config.T
    grad_W_gamma = -np.array([t_weigh[t]*w_t_curve(t) for t in range(T)])
    # grad_W_gamma = -np.array([config.time_weights[t] *
    #                     w_t.grad_eval(t, curve.eval_discrete(t)).reshape(2)
    #                     for t in range(config.T)])
    # (L(γ)∇W(γ)-W(γ)∇L(γ))/L(γ)²
    pos_gradient = (L_gamma*grad_W_gamma - W_gamma*grad_L_gamma)/L_gamma**2
    gradient_curve = curves.curve(pos_gradient)
    return gradient_curve

def after_optimization_sparsifier(current_measure, energy_curves=None):
    """ Trims a sparse measure by merging atoms that are too close.

    Given a measure composed of atoms, it will look for the atoms that are
    too close, and if is possible to maintain, or decrease, the energy of
    the measure by joining two atoms, it will do it.

    Parameters
    ----------
    current_measure : DGCG.curves.measure class
    energy_curves : numpy.ndarray, optional
        vector indicating the energy of the curves of the measure. To
        accelerate the comparisons.

    Returns
    -------
    DGCG.curves.measure class

    Notes
    -----
    This method is required because the quadratic optimization step is realized
    by an interior point method. Therefore, it is likely to find minimums in
    between two identical items instead of selecting one and discarding the
    other.
    """
    output_measure = copy.deepcopy(current_measure)
    id1 = 0
    id2 = 1
    num_curves = len(current_measure.curves)
    while id1 < num_curves:
        curve_1 = output_measure.curves[id1]
        while id2 < num_curves:
            curve_2 = output_measure.curves[id2]
            if (curve_1 - curve_2).H1_norm() < config.H1_tolerance:
                # if the curves are close, we have 3 alternatives to test
                weight_1 = output_measure.intensities[id1]
                weight_2 = output_measure.intensities[id2]
                measure_1 = copy.deepcopy(output_measure)
                measure_1.modify_intensity(id1, weight_1 + weight_2)
                measure_1.modify_intensity(id2, 0)
                measure_2 = copy.deepcopy(output_measure)
                measure_2.modify_intensity(id2, weight_1 + weight_2)
                measure_2.modify_intensity(id1, 0)
                energy_0 = output_measure.get_main_energy()
                energy_1 = measure_1.get_main_energy()
                energy_2 = measure_2.get_main_energy()
                min_energy = min([energy_0, energy_1, energy_2])
                if energy_1 == min_energy:
                    output_measure = measure_1
                    num_curves = num_curves - 1
                    id2 = id2 - 1
                elif energy_2 == min_energy:
                    output_measure = measure_2
                    num_curves = num_curves - 1
                    id1 = id1 - 1
                    id2 = num_curves
                else:
                    pass
            id2 = id2 + 1
        id1 = id1 + 1
        id2 = id1 + 1
    return output_measure

def solve_quadratic_program(current_measure):
    assert isinstance(current_measure, curves.measure)
    # Build the quadratic system of step 5 and then use some generic python
    # solver to get a solution.
    # Build matrix Q and vector b
    logger = config.logger
    # First, check that no curves are duplicated
    curves_list = current_measure.curves
    N = len(curves_list)
    Q = np.zeros((N, N), dtype=float)
    b = np.zeros(N)
    for i, curve in enumerate(curves_list):
        measure_i = curves.measure()
        measure_i.add(curve, 1)
        K_t_i = op.K_t_star_full(measure_i)
        b[i] = op.int_time_H_t_product(K_t_i, config.f_t)
        for j in range(i, N):
            measure_j = curves.measure()
            measure_j.add(curves_list[j], 1)
            K_t_j = op.K_t_star_full(measure_j)
            entry = op.int_time_H_t_product(K_t_i, K_t_j)
            Q[i, j] = entry
            Q[j, i] = entry
    QQ = Q
    try:
        Qc = cvxopt.matrix(QQ)
        bb = cvxopt.matrix(1 - b.reshape(-1, 1))
        G = cvxopt.matrix(-np.eye(N))
        h = cvxopt.matrix(np.zeros((N, 1)))
        sol = cvxopt.solvers.qp(Qc, bb, G, h)
        coefficients = np.array(sol['x']).reshape(-1)
    except Exception as e:
        print(e)
        print("Failed to use cvxopt, aborted.")
    # Incorporate as 0 coefficients those of the duplicates
    coefficients = list(coefficients)
    logger.status([1, 2, 2], coefficients)
    return curves_list, coefficients

def to_positive_semidefinite(Q):
    """ Takes a symmetric matrix and returns a positive semidefinite projection

    Parameters
    ----------
    Q : numpy.ndarray
        symmetric matrix

    Returns
    -------
    numpy.ndarray, symmetric positive semidefinite matrix.
    """
    min_eigval = 0
    eigval, eigvec = np.linalg.eigh(Q)
    if min(eigval) < min_eigval:
        # truncate
        print("Negative eigenvalues: ",
              [eig for eig in eigval if eig < min_eigval])
        eigval = np.maximum(eigval, min_eigval)
        # Recompute Q = VΣV^(-1)
        Q2 = np.linalg.solve(eigvec.T, np.diag(eigval)@eigvec.T).T
        print("PSD projection relative norm difference:",
              np.linalg.norm(Q-Q2)/np.linalg.norm(Q))
        return Q2
    return Q

def weight_optimization_step(current_measure):
    config.logger.status([1, 2, 1])
    # optimizes the coefficients for the current_measure
    # The energy_curves is a vector of energy useful for the trimming process
    assert isinstance(current_measure, curves.measure)
    curves_list, coefficients = solve_quadratic_program(current_measure)
    new_current_measure = curves.measure()
    for curve, intensity in zip(curves_list, coefficients):
        new_current_measure.add(curve, intensity)
    # Sparsifying step
    sparsier_measure = after_optimization_sparsifier(new_current_measure)
    return sparsier_measure

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
                     max_iter=config.g_flow_opt_in_between_iters):
    # We apply the gradient flow to simultaneously perturb the position of all
    # the current curves defining the measure.
    # Input: current_measure, measure type object.
    #        max_iter > 0 integer number of iterations to perform
    # Output: measure type object and finishing stepsize.
    logger = config.logger

    def full_gradient(current_measure):
        # Obtains the full gradient, which is an element in a H1 product space
        w_t = op.w_t(current_measure)
        curve_list = []
        for curve in current_measure.curves:
            curve_list.append(grad_F(curve, w_t))
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
            current_energy = current_measure.get_main_energy()
            if current_energy - new_main_energy > stepsize*m:
                break
            stepsize = stepsize*decrease_parameter
        return new_measure, stepsize
    # Descent implementation
    new_measure = current_measure
    # the initial step considered for the algorithm
    stepsize = init_step
    for iters in range(max_iter):
        new_measure, stepsize = \
                backtracking(new_measure, stepsize*1.2)
        logger.status([3, 0, 0], new_measure, stepsize, iters)
        if stepsize < limit_stepsize:
            break
    logger.status([3, 0, 1])
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


