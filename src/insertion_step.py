# Standard imports
import numpy as np
import copy
import code

# Local imports
from . import curves, config, insertion_mod
from . import operators as op
from . import optimization as opt

def insertion_step(current_measure):
    """ Insertion step + optimization step executed for a target sparse measure.

    -----------------------
    Inputs:
        current_measure (curves.measure class):
            representing the current iterate of the algorithm.
    Outputs:
        candidate_measure (curves.measure class):
            the measure obtained by applying the insertion step, followed by
            an optimization step
        exit_flag (integer):
            0 if no new inserted curve was found.
            1 if new inserted curve was found.
    """
    assert isinstance(current_measure, curves.measure)
    insertion_mod.initialize(current_measure)
    f_t = config.f_t
    logger = config.logger
    # compute the dual variable
    w_t = op.w_t(current_measure)
    # stationary_curves is a list of curves.curve objects, which are inserted in an
    # ordered fashion, with the order defined by their respectve F(γ) value,
    # pointed out in the energy_curves list.
    logger.status([1,1,0])
    # We use multistart descent to look for the global minimum. We obtain for free
    # a list of stationary curves.
    stationary_curves, energy_curves  = multistart_descent(current_measure)
    # log the found stationary curves
    logger.status([1,2,0], stationary_curves, energy_curves)
    # log the dual gap
    dual_gap = opt.dual_gap(current_measure, stationary_curves)
    logger.status([1,2,5], dual_gap)
    # Exit condition
    insertion_eps = config.insertion_eps
    if dual_gap < insertion_eps:
    #if energy_curves[1] >= -1 - insertion_eps:
        logger.status([1,2,4])
        exit_flag = 0 # the algorithm stops
        return current_measure, exit_flag
    else:
        # We proceed with the weight optimization step
        candidate_measure = curves.measure()
        for curve in current_measure.curves:
            candidate_measure.add(curve, 1)
        for curve in stationary_curves:
            candidate_measure.add(curve, 1)
        # Optimize the coefficients and create get a measure from them
        candidate_measure = opt.weight_optimization_step(candidate_measure,
                                                        energy_curves)
        exit_flag = 1
        return candidate_measure, exit_flag

def multistart_descent(current_measure):
    """ Uses multistart descent to search for the global minimizing curve.

    The multistart method corresponds to descent multiple randomly generated
    curves and to record the resulting stationary point of this descent
    expecting to find with this method the global minimizing curve.
    There are 2 main details:
        - To decrease the number of descents, this method routinely checks
        if the current descended curve is close to the already known ones.
        If so, it stops and discards the curve.
        - The descented curves are proposed by the insertion_mod module.
        It consists of: already known curves, crossover curves, random ones.
    --------------------
    Inputs:
        w_t (operators.w_t class):
            representing the dual variable of the problem at this iteration.
        current_measure (curves.measure class):
            the current iterate of the algorithm.
    Output:
        stationary_curves (list of curves.curve):
            list of curves, found stationary points of F(γ).
        energy_curves (list of floats, ordered):
            respective energy of the stationary_curves.
    Kwargs: None
    """
    logger = config.logger
    # needed initializations
    w_t = op.w_t(current_measure)
    energy_curves = []
    stationary_curves = []
    # load configuration parameters
    max_restarts = config.insertion_max_restarts
    insertion_min_restarts = config.insertion_min_restarts
    multistart_early_stop = config.multistart_early_stop # this is a function
    prop_max_iter = config.multistart_proposition_max_iter
    #
    min_energy = np.inf
    tries = 0
    while tries<= insertion_min_restarts or \
                      (tries <= max_restarts and
                       tries <= multistart_early_stop(len(energy_curves))):
        if len(energy_curves)>0:
            min_energy = min(energy_curves)
        logger.status([1,1,1], tries, stationary_curves)
        # The insertion module proposes curves to descend with negative energy
        proposed_energy = np.inf
        num_iter = 0
        while proposed_energy >= 0 and num_iter< prop_max_iter:
            new_curve = insertion_mod.propose(w_t, stationary_curves, energy_curves)
            proposed_energy = opt.F(new_curve, w_t)
            num_iter += 1
        if num_iter == prop_max_iter:
            raise Exception('Reached maximum number of tolerated proposed '+
                            'curves. Please inspect insertion_mod.propose '+
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
            # case 3: The descended curve is taking too much time to converge
            #         while not getting close enough to the taboo set.
            #         (this is if descent_soft_max_iter is reached)
            # case 3.1: If the value F(γ) is 0.9 close to the best known case,
            #           the descent continuous up to descent_max_iter is reached.
            # case 3.2: If the value F(γ) is not close enought to the best
            #           known case, the while loop is ended.
            close_to_known_set = False
            new_curve, stepsize = gradient_descent(new_curve, w_t,
                                                max_iter = inter_iters,
                                                init_step= stepsize)
            descent_iters += inter_iters
            new_curve_energy = opt.F(new_curve, w_t)
            logger.status([1,1,2])
            if is_close_to_stationaries(new_curve, new_curve_energy,
                                stationary_curves, energy_curves):
                # if the new_curve is too close to a stationary curve, break and discard
                logger.status([1,1,3])
                close_to_known_set = True
                if descent_iters == inter_iters:
                    # It just converged on the first set of iterations, does not
                    # count toward the iteration count
                    tries = tries - 1
                break
            if descent_iters >= descent_soft_max_iter:
                # check if the curve is getting somewhere good
                if new_curve_energy < min_energy*soft_max_threshold:
                    # It is going good
                    pass
                else:
                    # Just introduce it as it is into the stationary curve set
                    logger.status([1,1,4], new_curve_energy, min_energy)
                    # this is a way to exit simulating that the curve converged
                    stepsize = lim_stepsize/2
        if close_to_known_set == True:
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
                logger.status([1,1,5])
            elif stepsize <= lim_stepsize:
                logger.status([1,1,7], stationary_curves)
            else:
                raise Exception('Unexpected descent case')
        tries = tries+1
    return stationary_curves, energy_curves

def is_close_to_stationaries(new_curve, new_curve_energy,
                     stationary_curves, energy_curves) -> bool:
    """ Method to check if a given curve is close to the set of known stationary
    points.
    For acceleration, we use the known energy_curves values to only compare
    between curves with similar F(γ) values.
    """
    # We get the distance threshold to decide the curves are the same
    taboo_dist = config.multistart_taboo_dist
    # This distance is a guide to not compare with all the stationary_curves
    # and we compare only with those with less energy since any curve
    # that has more energy than the new_curve, is worse.
    energy_dist = config.multistart_energy_dist
    #
    lower_index = np.searchsorted(energy_curves,
                              new_curve_energy -energy_dist, side='left')
    upper_index = np.searchsorted(energy_curves,
                              new_curve_energy, side='right')
    #
    for idx in range(lower_index, upper_index):
        forbidden_curve = stationary_curves[idx]
        if (new_curve - forbidden_curve).H1_norm() < taboo_dist:
            return True
    return False

def gradient_descent(curve, w_t, max_iter  = None, init_step = None,
                                 limit_stepsize = None):
    """ Gradient descent operator G.

    Implemented gradient descent operator G of subroutine 1. In considers
    Armijo stepsize rule with backtracking.
    This method assumes that the given starting curve γ satisfies
    F(γ) < 0.
    """
    assert isinstance(curve, curves.curve) and isinstance(w_t, op.w_t)
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
        control_parameter  = 1e-10
        gradient = opt.grad_F(curve, w_t)
        m = control_parameter*gradient.H1_norm()
        while stepsize >= limit_stepsize:
            new_curve = curve-stepsize*gradient
            energy_new_curve = opt.F(new_curve, w_t)
            if energy_curve - energy_new_curve  > stepsize*m:
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
        logger.status([1,0,3], i, energy_curve, stepsize)
        new_curve, energy_curve, stepsize =  \
            backtracking(new_curve, energy_curve, stepsize*1.2)
        if stepsize < limit_stepsize:
            break
    logger.status([1,0,4])
    return new_curve, stepsize
