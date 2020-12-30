# Standard imports
import numpy as np
import copy
import code

# Local imports
from . import classes, config, insertion_mod
from . import optimization as opt


def insertion_step(current_measure):
    """Insertion step & optimization step executed on a target measure.

    Parameters
    ----------
    current_measure : :py:class:`src.classes.measure`
        Target measure to apply the inserion + optimization step

    Returns
    -------
    new_measure : :py:class:`src.classes.measure`
    exit_flag : int
        0 if no new inserted curve was found. 1 else.
    """
    assert isinstance(current_measure, classes.measure)
    insertion_mod.initialize(current_measure)
    logger = config.logger
    # stationary_curves is a list of classes.curve objects, which are inserted
    # in an ordered fashion, with the order defined by their respectve F(γ)
    # value, pointed out in the energy_curves list.
    logger.status([1, 1, 0])
    # We use multistart descent to look for the global minimum. We obtain for
    # free a list of stationary curves.
    stationary_curves, energy_curves = multistart_descent(current_measure)
    # log the found stationary curves
    logger.status([1, 2, 0], stationary_curves, energy_curves)
    # log the dual gap
    dual_gap = opt.dual_gap(current_measure, stationary_curves)
    logger.status([1, 2, 5], dual_gap)
    # Exit condition
    insertion_eps = config.insertion_eps
    if dual_gap < 0:
        print('Somehow dual gap negative, something must be wrong')
        print('Likely the TOL value is too small these are rounding errors')
    if dual_gap < insertion_eps:
        logger.status([1, 2, 4])
        exit_flag = 0  # the algorithm stops
        return current_measure, exit_flag
    else:
        # We proceed with the weight optimization step
        max_curve_num = config.curves_list_length_lim
        max_stationary = max_curve_num - len(current_measure.curves)
        max_stationary = max(max_stationary, config.curves_list_length_min)
        candidate_measure = classes.measure()
        for curve in current_measure.curves:
            candidate_measure.add(curve, 1)
        for curve in stationary_curves[:max_stationary]:
            candidate_measure.add(curve, 1)
        # Optimize the coefficients and create get a measure from them
        candidate_measure = opt.weight_optimization_step(candidate_measure)
        exit_flag = 1
        return candidate_measure, exit_flag

def multistart_descent(current_measure):
    """ Uses multistart descent to search for the global minimizing curve.

    The multistart method corresponds to descent multiple randomly generated
    curves and to record the resulting stationary point of this descent
    expecting to find with this method the global minimizing curve.
    Some details:

    Parameters
    ----------
    current_measure : :py:class:`src.classes.measure`
            the current iterate of the algorithm.

    Returns
    -------
    stationary_curves : list[:py:class:`src.classes.curve`]
        list of the found stationary points of the insertion step problem.
    energy_curves : numpy.ndarray
        respective energy of the found stationary_curves, sorted in ascending
        order.

    Notes
    -----
    - To decrease the number of descents, this method routinely checks
    if the current descended curve is close to the already known ones.
    If so, it stops and discards the curve.
    - The descented curves are proposed by :py:meth:`src.insertion_mod.propose`
    It consists of: already known curves, crossover curves, random ones.
    - If a crossover curve gets too close to a stationary curve earlier
    than the first check, it is not counted as an attempt.
    """
    logger = config.logger
    # needed initializations
    w_t = classes.dual_variable(current_measure)
    energy_curves = []
    stationary_curves = []
    # load configuration parameters
    max_restarts = config.insertion_max_restarts
    insertion_min_restarts = config.insertion_min_restarts
    multistart_early_stop = config.multistart_early_stop  # this is a function
    prop_max_iter = config.multistart_proposition_max_iter
    #
    max_discarded_tries = config.multistart_max_discarded_tries
    max_discarded_counter = 0
    #
    min_energy = np.inf
    tries = 0
    while tries <= insertion_min_restarts or \
        tries <= min(max_restarts,
                     multistart_early_stop(tries, len(energy_curves))):
        if len(energy_curves) > 0:
            min_energy = min(energy_curves)
        logger.status([1, 1, 1], tries, stationary_curves)
        # The insertion module proposes curves to descend with negative energy
        proposed_energy = np.inf
        num_iter = 0
        while proposed_energy >= 0 and num_iter < prop_max_iter:
            new_curve = insertion_mod.propose(w_t, stationary_curves,
                                              energy_curves)
            proposed_energy = opt.F(new_curve, w_t)
            num_iter += 1
        if num_iter == prop_max_iter:
            raise Exception('Reached maximum number of tolerated proposed ' +
                            'curves. Please inspect insertion_mod.propose ' +
                            'method')
        # descent the curve
        descent_iters = 0
        descent_max_iter = config.multistart_descent_max_iter
        descent_soft_max_iter = config.multistart_descent_soft_max_iter
        soft_max_threshold = config.multistart_descent_soft_max_threshold
        stepsize = config.multistart_descent_init_step
        lim_stepsize = config.multistart_descent_limit_stepsize
        inter_iters = config.multistart_inter_iteration_checkup
        #
        while descent_iters < descent_max_iter and stepsize > lim_stepsize:
            # This while-loop applies the gradient descent on curves,
            # while simultaneously it checks in intermediates steps if
            # certain conditions are satisfied. These are the possible cases:
            # case 1: A stationary point is found. This is captured when the
            #         stepsize goes below lim_stepsize.
            # case 2: The descended curve got at some point close to the
            #         stationary set. The while breaks.
            # case 2.2: If this curve gets too close before the first check,
            #           it is not counted as an attempt.
            # case 3: The descended curve is taking too much time to converge
            #         while not getting close enough to the taboo set.
            #         (this is if descent_soft_max_iter is reached)
            # case 3.1: If the value F(γ) is 0.9 close to the best known case,
            #           the descent continuous up to descent_max_iter is
            #           reached.
            # case 3.2: If the value F(γ) is not close enought to the best
            #           known case, the while loop is ended.
            close_to_known_set = False
            new_curve, stepsize = gradient_descent(new_curve, w_t,
                                                   max_iter=inter_iters,
                                                   init_step=stepsize)
            descent_iters += inter_iters
            new_curve_energy = opt.F(new_curve, w_t)
            logger.status([1, 1, 2])
            if is_close_to_stationaries(new_curve, new_curve_energy,
                                        stationary_curves, energy_curves):
                # if the new_curve is too close to a stationary curve, break
                # and discard
                logger.status([1, 1, 3])
                close_to_known_set = True
                if descent_iters == inter_iters:
                    # It just converged on the first set of iterations, does
                    # not count toward the iteration count
                    max_discarded_counter += 1
                    if max_discarded_counter >= max_discarded_tries:
                        print("""WARNING: Most of the proposed curves are
                              converging faster than the first checkout, making
                              them not count towards the numbe of tries and
                              potentially leading towards an infinite loop.
                              Please reconsider decreasing the value of
                              config.multistart_inter_interation_checkup
                              or increasing the value of
                              config.multistart_taboo_dist""")
                        max_discarded_counter = 0
                    else:
                        tries = tries - 1
                else:
                    max_discarded_counter = 0
                break
            if descent_iters >= descent_soft_max_iter:
                # check if the curve is getting somewhere good
                if new_curve_energy < min_energy*soft_max_threshold:
                    # It is going good
                    pass
                else:
                    # Just introduce it as it is into the stationary curve set
                    logger.status([1, 1, 4], new_curve_energy, min_energy)
                    # this is a way to exit simulating that the curve converged
                    stepsize = lim_stepsize/2
        if close_to_known_set:
            pass
        else:
            # In all the other cases, the descended curve is inserted in
            # the taboo set.
            # We insert them in a sorted fashion
            insert_index = np.searchsorted(energy_curves, new_curve_energy)
            energy_curves.insert(insert_index, new_curve_energy)
            stationary_curves.insert(insert_index, new_curve)
            # the insertion mod needs to know the order of the curves
            insertion_mod.update_crossover_memory(insert_index)
            if descent_iters >= descent_max_iter:
                # Reached maximum of iterations, added to stationary curves set
                logger.status([1, 1, 5])
            elif stepsize <= lim_stepsize:
                logger.status([1, 1, 7], stationary_curves)
            else:
                raise Exception('Unexpected descent case')
        tries = tries+1
    return stationary_curves, energy_curves


def is_close_to_stationaries(new_curve, new_curve_energy,
                             stationary_curves, energy_curves) -> bool:
    """Checks if a given curve is close to the set of found stationary curves.

    The distance is measured with the :math:`H^1` norm, and the threshold is
    set by ``config.multistart_taboo_dist``.

    Parameters
    ----------
    new_curve : :py:class:`src.classes.curve`
        Curve to check if it is close to the stationary set
    new_curve_energy : float
        Energy of the curve to check
    stationary_curves : list[:py:class:`src.classes.curve`]
        List of found stationary curves
    energy_curves : numpy.ndarray
        Energies of the found stationary curves sorted in ascendent order.

    Notes
    -----
    The energy_curves are used to accelerate the comparisons. To avoid
    with the whole set of found stationary curves.
    """
    # We get the distance threshold to decide the curves are the same
    taboo_dist = config.multistart_taboo_dist
    # This distance is a guide to not compare with all the stationary_curves
    # and we compare only with those with less energy since any curve
    # that has more energy than the new_curve, is worse.
    energy_dist = config.multistart_energy_dist
    #
    lower_index = np.searchsorted(energy_curves,
                                  new_curve_energy - energy_dist, side='left')
    upper_index = np.searchsorted(energy_curves, new_curve_energy,
                                  side='right')
    #
    for idx in range(lower_index, upper_index):
        forbidden_curve = stationary_curves[idx]
        if (new_curve - forbidden_curve).H1_norm() < taboo_dist:
            return True
    return False


def gradient_descent(curve, w_t, max_iter=None, init_step=None,
                     limit_stepsize=None):
    """Applies the gradient descent to an input curve.

    The function to minimize F(γ) is defined via the dual variable. The
    Applied gradient descent is the Armijo with backtracking, with stopping
    condition reached when the stepsize reaches a predefined value.

    Parameters
    ----------
    curve : :py:class:`src.classes.curve`
        Curve to be descended.
    w_t : :py:class:`src.classes.dual_variable`
        Dual variable associated to the current iterate.
    max_iter : int, optional
        A bound on the number of iterations. Defaults to
        ``config.multistart_descent_max_iter``.
    init_step : float, optional
        Defines the initial step of the descent method. Defaults to
        ``config.multistart_descent_init_step``.
    limit_stepsize : float, optional
        The stopping condition for the gradient descent. Defaults to
        ``config.multistart_descent_limit_stepsize``

    Returns
    -------
    :py:class:`src.classes.curve`

    Notes
    -----
    As described in the paper, the gradient descent assumes that the input
    curve has negative energy: F(γ) < 0.
    """
    assert isinstance(curve, classes.curve) and \
           isinstance(w_t, classes.dual_variable)
    # Applies the gradient descent algorithm
    # inherited parameters
    if max_iter is None:
        max_iter = config.multistart_descent_max_iter
    if init_step is None:
        init_step = config.multistart_descent_init_step
    if limit_stepsize is None:
        limit_stepsize = config.multistart_descent_limit_stepsize
    logger = config.logger
    # Armijo + backtracking implementation

    def backtracking(curve, energy_curve, stepsize):
        decrease_parameter = 0.8
        control_parameter = 1e-10
        gradient = opt.grad_F(curve, w_t)
        m = control_parameter*gradient.H1_norm()
        while stepsize >= limit_stepsize:
            new_curve = curve-stepsize*gradient
            energy_new_curve = opt.F(new_curve, w_t)
            if energy_curve - energy_new_curve > stepsize*m:
                break
            stepsize = stepsize*decrease_parameter
        return new_curve, energy_new_curve, stepsize
    # Descent implementation
    new_curve = curve
    energy_curve = opt.F(new_curve, w_t)
    assert isinstance(energy_curve, float)
    # the initial step considered for the algorithm
    stepsize = init_step
    for i in range(max_iter):
        logger.status([1, 0, 3], i, energy_curve, stepsize)
        new_curve, energy_curve, stepsize =  \
            backtracking(new_curve, energy_curve, stepsize*1.2)
        if stepsize < limit_stepsize:
            break
    logger.status([1, 0, 4])
    return new_curve, stepsize
