# Standard imports
import copy
import numpy as np

# Not so standard imports
import cvxopt

# Local imports
from . import classes, config
from . import operators as op

# Solver parameters
cvxopt.solvers.options['reltol'] = config.CVXOPT_TOL
cvxopt.solvers.options['abstol'] = config.CVXOPT_TOL
cvxopt.solvers.options['show_progress'] = False  # to silence solver


def F(curve, w_t):
    """ The F(γ) operator, minimization target in the insertion step.

    Parameters
    ----------
    curve : :py:class:`src.classes.curve`
        Curve γ where the F operator is evaluated.
    w_t : :py:class:`src.classes.dual_variable`
        Dual variable that defines the F operator.

    Returns
    -------
    float


    Notes
    -----
    The F operator is defined via the dual variable as

    .. math::
        F(\\gamma) = -\\frac{a_{\\gamma}}{T+1} \\sum_{t=0}^T w_t(\\gamma(t))

    with :math:`a_{\\gamma} =
    1/(\\frac{\\beta}{2}\\int_0^1 ||\\dot \\gamma(t)||^2dt + \\alpha)`
    """
    assert isinstance(curve, classes.curve) and \
           isinstance(w_t, classes.dual_variable)
    return -curve.integrate_against(w_t)/curve.energy()

def grad_F(curve, w_t):
    """ The gradient of the F operator, ∇F(γ).

    Parameters
    ----------
    curve : :py:class:`src.classes.curve`
        Curve γ where the F operator is evaluated.
    w_t : :py:class:`src.classes.dual_variable`
        Dual variable that defines the F operator.

    Returns
    -------
    :py:class:`src.classes.curve`

    Notes
    -----
    The F operator is defined on the Hilbert space of curves, therefore the
    gradient should be a curve.
    """
    assert isinstance(curve, classes.curve) and \
           isinstance(w_t, classes.dual_variable)
    # F = W/L
    L_gamma = curve.energy()
    W_gamma = -curve.integrate_against(w_t)
    # ∇L(γ) computation
    diff_positions = np.diff(curve.spatial_points, axis=0)  # γ_{i+1}-γ_{i} (T-1)x2 array
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
    # (L(γ)∇W(γ)-W(γ)∇L(γ))/L(γ)²
    pos_gradient = (L_gamma*grad_W_gamma - W_gamma*grad_L_gamma)/L_gamma**2
    gradient_curve = classes.curve(pos_gradient)
    return gradient_curve


def after_optimization_sparsifier(current_measure):
    """ Trims a sparse measure by merging atoms that are too close.

    Given a measure composed of atoms, it will look for the atoms that are
    too close, and if is possible to maintain, or decrease, the energy of
    the measure by joining two atoms, it will do it.

    Parameters
    ----------
    current_measure : :py:class:`src.classes.measure`
        Target measure to trim.

    Returns
    -------
    DGCG.classes.measure class

    Notes
    -----
    This method is required because the quadratic optimization step is realized
    by an interior point method. Therefore, in the case that there are repeated
    (or very close to repeated) atoms in the current measure, the quadratic
    optimization step can give positive weights to both of them.

    This is not desirable, since besides incrementing the computing power for
    the sliding step, we would prefer each atom numerically represented only
    once.
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
                weight_1 = output_measure.weights[id1]
                weight_2 = output_measure.weights[id2]
                measure_1 = copy.deepcopy(output_measure)
                measure_1.modify_weight(id1, weight_1 + weight_2)
                measure_1.modify_weight(id2, 0)
                measure_2 = copy.deepcopy(output_measure)
                measure_2.modify_weight(id2, weight_1 + weight_2)
                measure_2.modify_weight(id1, 0)
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
    """Compute optimal weights for a given measure.

    Parameters
    ----------
    current_measure : :py:class:`src.classes.measure`.

    Returns
    -------
    list[:py:class:`src.classes.curve`]
        List of curves/atoms with non-zero weights.
    list[float]
        List of positive optimal weights.

    Notes
    -----
    The solved problem is
 
    .. math::
        \\min_{(c_1,c_2, ... )}
        T_{\\alpha, \\beta}\\left( \\sum_{j} c_j \\mu_{\\gamma_j}\\right)

    Where :math:`T_{\\alpha, \\beta}` is the main energy to minimize
    :py:meth:`src.operators.main_energy` and :math:`\\mu_{\\gamma_j}`
    represents the atoms of the current measure.

    This quadratic optimization problem is solved using the `CVXOPT solver
    <https://cvxopt.org/>`_.
    """
    assert isinstance(current_measure, classes.measure)
    # Build matrix Q and vector b
    logger = config.logger
    # First, check that no curves are duplicated
    curves_list = current_measure.curves
    N = len(curves_list)
    Q = np.zeros((N, N), dtype=float)
    b = np.zeros(N)
    for i, curve in enumerate(curves_list):
        measure_i = classes.measure()
        measure_i.add(curve, 1)
        K_t_i = op.K_t_star_full(measure_i)
        b[i] = op.int_time_H_t_product(K_t_i, config.f_t)
        for j in range(i, N):
            measure_j = classes.measure()
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


def weight_optimization_step(current_measure):
    """Applies the weight optimization step to target measure.

    Both optimizes the weights and trims the resulting measure.

    Parameters
    ----------
    current_measure : :py:class:`src.classes.measure`
        Target sparse dynamic measure.

    Returns
    -------
    :py:class:`src.classes.curves`

    Notes
    -----
    To find the optimal weights, it uses
    :py:meth:`src.optimization.solve_quadratic_program`, to trim
    :py:meth:`src.optimization.after_optimization_sparsifier`.
    """
    config.logger.status([1, 2, 1])
    # optimizes the coefficients for the current_measure
    # The energy_curves is a vector of energy useful for the trimming process
    assert isinstance(current_measure, classes.measure)
    curves_list, coefficients = solve_quadratic_program(current_measure)
    new_current_measure = classes.measure()
    for curve, intensity in zip(curves_list, coefficients):
        new_current_measure.add(curve, intensity)
    # Sparsifying step
    sparsier_measure = after_optimization_sparsifier(new_current_measure)
    return sparsier_measure

def slide_and_optimize(current_measure):
    """Applies alternatedly the sliding and optimization step to measure.

    The sliding step consists in fixing the weights of the measure and then,
    as a function of the curves, use the gradient descent to minimize the
    target energy. The optimization step consists in fixing the curves and
    then optimize the weights to minimize the target energy.

    This method alternates between sliding a certain number of times, and then
    optimizating the weights. It stops when it reaches the convergence critera,
    or reaches a maximal number of iterations.

    Parameters
    ----------
    current_measure : :py:class:`src.classes.measure`
        Target measure to slide and optimize

    Returns
    -------
    :py:class:`src.classes.measure`

    Notes
    -----
    To control the different parameters that define this method (alternation
    rate, convergence critera, etc) see
    :py:data:`src.config.slide_opt_max_iter`,
    :py:data:`src.config.slide_opt_in_between_iters`,
    :py:data:`src.config.slide_init_step`,
    :py:data:`src.config.slide_limit_stepsize`
    """
    assert isinstance(current_measure, classes.measure)
    stepsize = config.slide_init_step
    total_iterations = 0
    while total_iterations <= config.slide_opt_max_iter:
        current_measure, stepsize, iters = gradient_descent(current_measure,
                                                            stepsize)
        total_iterations += config.slide_opt_in_between_iters
        current_measure = weight_optimization_step(current_measure)
        if stepsize < config.slide_limit_stepsize:
            # The gradient flow converged, but since the coefficients got
            # optimized, it is required to restart the gradient flow.
            stepsize = np.sqrt(config.slide_limit_stepsize)
        if iters == 0:
            # The current measure is already optimal, therefore, there is no
            # need to keep iterating
            break
    return current_measure


def gradient_descent(current_measure, init_step,
                     max_iter=config.slide_opt_in_between_iters):
    """Applies the gradient descent to the curves that define the measure.

    This method descends a the function that takes a fixed number of
    of curves and maps it to the main energy to minimize applied to the measure
    with these curves as atoms and fixed weights. It uses an Armijo with
    backtracking descent.

    Parameters
    ----------
    current_measure : :py:class:`src.classes.measure`
        Measure defining the starting curves and fixed weights from which to
        descend.
    init_step : float
        The initial step of the gradient descent.
    max_iter : int, optional
        The maximum number of iterations. Default
        :py:data:`src.config.slide_opt_it_between_iters`

    Returns
    -------
    new_measure : :py:class:`src.classes.measure`
        Resulting measure from the descent process.
    stepsize : float
        The final reached stepsize.
    iter : int
        The number of used iterations to converge.
    """

    # Output: measure type object and finishing stepsize.
    logger = config.logger

    def full_gradient(current_measure):
        # Obtains the full gradient, which is an element in a H1 product space
        w_t = classes.dual_variable(current_measure)
        curve_list = []
        for curve in current_measure.curves:
            curve_list.append(grad_F(curve, w_t))
        return classes.curve_product(curve_list, current_measure.weights)
    # Stop when stepsize get smaller than
    limit_stepsize = config.slide_limit_stepsize

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
    """ Dual gap of the current measure.

    The dual computed using a supplied set of stationary curves obtained
    from the multistart gradient descent
    :py:meth:`src.insertion_step.multistart_descent`.

    Parameters
    ----------
    current_measure : :py:class:`src.classes.measure`
        Current measure to compute the dual gap.
    stationary_curves : list[:py:class:`src.classes.curve`]
        Set of stationary curves, ordered incrementally by their F(γ) value.

    Returns
    -------
    float

    Notes
    -----
    It is assumed that the first element of the stationary curves is the
    best one and it satisfies
    :math:`F(\\gamma) \\leq -1`. This is ensured since the multistart gradient
    descent descents the curves that are known from the last iterate, and the
    theory tells us that those curves satisfy :math:`F(\\gamma) = -1`.

    Therefore, according to the theory, to compute the dual gap we can use
    the formula

    .. math::
        \\text{dual gap} = \\frac{M_0}{2} ( |<w_t, \\rho_{\\gamma^*}
        >_{\\mathcal{M}; \\mathcal{C}}|^2 - 1) = \\frac{M_0}{2} \\left(\\left(
        \\frac{a_{\\gamma}}{T+1} \\sum_{t=0}^{T} w_t(\\gamma(t))\\right)^2 -1
        \\right)

    With :math:`a_{\\gamma} = 1/(\\frac{\\beta}{2} \\int_0^1 ||\\dot \\gamma(t)
    ||^2 dt + \\alpha)` and :math:`M_0 = T_{\\alpha, \\beta}(0)`, the main
    energy :py:meth:`src.operators.main_energy` evaluated in the zero measure.
    """
    # Compute the dual variable
    w_t = classes.dual_variable(current_measure)
    # Compute the constant (??)
    M_0 = op.int_time_H_t_product(config.f_t, config.f_t)/2
    # Extract the global minimizer
    insertion_step_minimizer = stationary_curves[0]
    # Build a measure with the global minimizer
    compare_measure = classes.measure()
    compare_measure.add(insertion_step_minimizer, 1)
    # Formula
    return M_0*(compare_measure.integrate_against(w_t)**2 - 1)/2


