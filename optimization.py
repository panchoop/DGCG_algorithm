import numpy as np
import scipy
from scipy.integrate import ode
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
import copy
import code

import cvxopt
import curves
import operators as op
import config
import insertion_mod

# Solver parameters
cvxopt.solvers.options['reltol']=1e-16
cvxopt.solvers.options['abstol']=1e-16

# Global fixed parameters
alpha = config.alpha
beta = config.beta

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
        max_iter = config.step3_descent_max_iter
    if init_step is None:
        init_step = config.step3_descent_init_step
    if limit_stepsize is None:
        limit_stepsize = config.step3_descent_limit_stepsize
    logger = config.logger
    # Armijo + backtracking implementation
    def backtracking(curve, energy_curve, stepsize):
        decrease_parameter = 0.8
        control_parameter  = 1e-10
        gradient = grad_F(curve, w_t)
        m = control_parameter*gradient.H1_norm()
        while stepsize >= limit_stepsize:
            new_curve = curve-stepsize*gradient
            energy_new_curve = F(new_curve, w_t)
            if energy_curve - energy_new_curve  > stepsize*m:
                break
            stepsize = stepsize*decrease_parameter
        return new_curve, energy_new_curve, stepsize
    # Descent implementation
    new_curve = curve
    energy_curve = F(new_curve, w_t)
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

def taboo_search(w_t, tabu_curves, energy_curves, current_measure,
                     min_energy_threshold = -1, current_tries = 0 ):
    assert isinstance(w_t, op.w_t) and isinstance(tabu_curves, list) \
            and isinstance(energy_curves, list)
    # Uses taboo search to find local minimum curves.
    # A random curve will be generated with random_insertion(w_t)
    # Then it will descend it until reaching a local minimum.
    # The algorithm takes care to eliminate the repeated local minimums.
    # It will try a minimum preset number of times, then two things happen:
        # If of those obtained curves has lower energy than the input threshold
        #    it will output the whole set of found local minimums. 
        # If none of the found curves has low enough energy, it will keep
        #    iterating for until finding a curve with lower energy, or until
        #    reaching a maximum number of iterations. In this case, it will
        #    just output the set of all the found local minimums.
    # INPUT: w_t is dual variable type, see operators.K_t, operators.grad_K_t
    #        tabu_curves is a list of curve type objects.
    # OUTPUT: new_curve is a curve type object
    #         new_curve_energy < 0 is a scalar.
    logger = config.logger
    def is_close_to_tabu(new_curve, new_curve_energy,
                         tabu_curves, energy_curves):
        # Method to indicate if new_curve is close to some curve in tabu_curves
        # new_curve is a curves.curve type object
        # new_curve_energy corresponds to F(γ), with γ the new_curve.
        # tabu_curves is a list of curves.curve type objects
        # energy_curves is a list of floats with the values <w_t, \rho_\gamma>
        #               of the respective tabu curves

        # We get the distance threshold to decide the curves are the same
        tabu_dist = config.step3_tabu_dist
        # This distance is a guide to not compare with all the tabu_curves
        # and we compare only with those with less energy since any curve
        # that has more energy than the new_curve, is worse.
        energy_dist = config.step3_energy_dist
        #
        lower_index = np.searchsorted(energy_curves,
                                  new_curve_energy -energy_dist, side='left')
        upper_index = np.searchsorted(energy_curves,
                                  new_curve_energy, side='right')
        #
        for idx in range(lower_index, upper_index):
            forbidden_curve = tabu_curves[idx]
            if (new_curve - forbidden_curve).H1_norm() < tabu_dist:
                return True
        return False
    # iterate looking for stationary curves
    min_number_of_attempts = config.step3_min_attempts_to_find_better_curve
    max_number_of_attempts = config.step3_max_attempts_to_find_better_curve
    max_number_of_failures = config.step3_max_number_of_failures
    tries = current_tries
    while tries <= max_number_of_attempts:
        if len(energy_curves)>0:
            min_energy = min(energy_curves)
        else:
            min_energy = -1
        if tries >= min_number_of_attempts \
                                and min_energy <min_energy_threshold:
            # Found a good curve candidate
            logger.status([1,1,6], energy_curves, min_energy_threshold)
            break
        logger.status([1,1,1], tries, tabu_curves)
        # The insertion module proposes curves to descend with negative energy
        proposed_energy = np.inf
        max_iter = 10000
        num_iter = 0
        while proposed_energy >= 0 and num_iter<max_iter:
            new_curve = insertion_mod.propose(w_t, tabu_curves, energy_curves)
            proposed_energy = F(new_curve, w_t)
            num_iter += 1
        if num_iter == max_iter:
            raise Exception('Reached maximum number of tolerated proposed '+
                            'curves. Please inspect insertion_mod.propose '+
                            'method')
        # descent the curve
        descent_iters = 0
        descent_max_iter = config.step3_descent_max_iter
        descent_soft_max_iter = config.step3_descent_soft_max_iter
        stepsize = config.step3_descent_init_step
        lim_stepsize = config.step3_descent_limit_stepsize
        inter_iters = config.step3_tabu_in_between_iteration_condition_checkup
        while descent_iters < descent_max_iter and stepsize > lim_stepsize:
            # This while-loop applies the gradient descent on curves,
            # while simultaneously it checks in intermediates steps if 
            # certain conditions are satisfied. These are the possible cases:
            # case 1: A stationary point is found. This is captured when the
            #         stepsize goes below lim_stepsize. 
            # case 2: The descended curve got at some point close to the tabu 
            #         set. The while breaks.
            # case 3: The descended curve is taking too much time to converge
            #         while not getting close enough to the taboo set.
            #         (this is if descent_soft_max_iter is reached)
            # case 3.1: If the value F(γ) is 0.9 close to the best known case,
            #           the descent continuous up to descent_max_iter is reached.
            # case 3.2: If the value F(γ) is not close enought to the best
            #           known case, the while loop is ended.
            close_to_tabu_flag = False
            new_curve, stepsize = gradient_descent(new_curve, w_t,
                                                max_iter = inter_iters,
                                                init_step= stepsize)
            descent_iters += inter_iters
            new_curve_energy = F(new_curve, w_t)
            logger.status([1,1,2])
            if is_close_to_tabu(new_curve, new_curve_energy,
                                tabu_curves, energy_curves):
                # if the new_curve is too close to a tabu_curve, break and discard
                logger.status([1,1,3])
                close_to_tabu_flag = True
                if descent_iters == inter_iters:
                    # It just converged on the first set of iterations, does not
                    # count toward the iteration count
                    tries = tries - 1
                break
            if descent_iters >= descent_soft_max_iter:
                # check if the curve is getting somewhere good
                if new_curve_energy < min_energy*0.9:
                    # It is going good
                    pass
                else:
                    # Just introduce it as it is into the tabu curve
                    logger.status([1,1,4], new_curve_energy, min_energy)
                    # this is a way to exit simulating that the curve converged
                    stepsize = lim_stepsize/2
        if close_to_tabu_flag == True:
            pass
        else:
            # In all the other cases, the descended curve is inserted in 
            # the taboo set.
            # We insert them in a sorted fashion
            insert_index = np.searchsorted(energy_curves, new_curve_energy)
            energy_curves.insert(insert_index, new_curve_energy)
            tabu_curves.insert(insert_index, new_curve)
            # the insertion mod needs to know the order of the curves
            insertion_mod.update_crossover_memory(insert_index)
            if descent_iters >= descent_max_iter:
                # Reached maximum of iterations, added to tabu curves set
                logger.status([1,1,5])
            elif stepsize <= lim_stepsize:
                logger.status([1,1,7], tabu_curves)
            else:
                raise Exception('Unexpected descent case')
        tries = tries+1
    return tabu_curves, energy_curves, tries

def insertion_step(current_measure):
    assert isinstance(current_measure, curves.measure)
    # Insertion step of the algorithm. Given a current measure, inserts 
    # a measure that attempts to solve the linearized minimization problem.
    # The algorithm will attempt to insert random curves
    # Then will descend them until converging to some local minimum
    # Using Tabu search, will attempt to find the global minimum of the problem.
    # Predefined things to start, <+write_description_here+>
    insertion_mod.initialize(current_measure)
    f_t = config.f_t
    logger = config.logger
    # Get the dual variable
    w_t = op.w_t(current_measure)
    # Tabu iterations seeking for a global minimum
    number_of_tries = 0
    min_energy_threshold = -1
    # tabu_curves is a list of measure.curve objects, which are inserted in an
    # ordered fashion, with the order defined by their respectve F(γ) value,
    # pointed out in the energy_curves list.
    tabu_curves = []
    energy_curves = []
    while number_of_tries < config.step3_max_attempts_to_find_better_curve:
        # taboo search, we obtain a new curve with lower energy
        logger.status([1,1,0])
        tabu_curves, energy_curves, tries = \
                taboo_search(w_t, tabu_curves, energy_curves,
                                   current_measure, min_energy_threshold,
                                   number_of_tries)
        number_of_tries = tries
        # Add the lowest energy curve to the current_measure and optimize
        # coefficients
        logger.status([1,2,0], tabu_curves, energy_curves)
        # Optimize the coefficients and create get a measure from them
        candidate_measure = curves.measure()
        for curve in current_measure.curves:
            candidate_measure.add(curve, 1)
        for curve in tabu_curves:
            candidate_measure.add(curve, 1)
        candidate_measure = coefficient_optimization_step(candidate_measure,
                                                        energy_curves)
        # Test if the energy got decreased
        if candidate_measure.get_main_energy()+config.energy_change_tolerance \
                                          <  current_measure.get_main_energy():
            # The new curve is accepted as a good step
            logger.status([1,2,4])
            min_energy_threshold = min(energy_curves)
            break
        else:
            # The new curve did not decrease enough 
            logger.status([1,2,3], candidate_measure.get_main_energy(),
                                   current_measure.get_main_energy())
            min_energy_threshold = min(energy_curves)
    if number_of_tries >= config.step3_max_attempts_to_find_better_curve:
        print("The algorithm has reached the maximum number of tries for the"+
              " tabu search to find a better candidate, the current measure is"+
              "declared to be the optimal solution")
        return None
    logger.status([1,2,5], current_measure, tabu_curves, energy_curves) # Dual gap 
    current_measure = candidate_measure
#    for i in range(len(current_measure.curves)-1, -1, -1):
#        current_measure.modify_intensity(i, new_coefficients[i])
    return current_measure

def measure_trimming(current_measure, energy_curves = None, H1_tol_factor = 1):
    # Step to ensure that the quadratic optimization does not fail
    # the quadratic optimization fails when the correlation matrix Q is not
    # positive definite. A first step is to eliminate accidentally duplicated
    # curves. A second step is to trim too low energy curves in the case 
    # there are too many curves
    # Two cases, if the curves energies are provided or not
    # The H1_tol_factor is a number >= 1 that is increased in the case
    # in which the quadratic optimization fails and it is required to be more
    # lax on what is "to be the same curve".
    # The vector of energy_curves concerns only the Tabu curves, not the
    # current_measure one's. The number of tabu_curves and current_measure's
    # ones is implicitly given with the length difference between the 
    # energy_curves vector and the number of current_measure curves.
    # OUTPUT: a list of curves, representing the measure
    curves_list = copy.deepcopy(current_measure.curves)
    if energy_curves == None:
        duplicates_idx = []
        for i, curve1 in enumerate(curves_list):
            for curve2 in curves_list[i+1:]:
                if (curve1-curve2).H1_norm() < config.H1_tolerance*H1_tol_factor:
                    duplicates_idx.append(i)
                    break
        # take out the duplicated curves
        for i in reversed(duplicates_idx):
            curves_list.pop(i)
    else:
        # Get the number of current curves and tabu curves
        N_current_curves = len(curves_list)-len(energy_curves)
        N_tabu_curves = len(energy_curves)
        # separate curves
        current_curves = curves_list[0:N_current_curves]
        tabu_curves = curves_list[N_current_curves:]
        # Order the tabu curves
        sort_idx = np.argsort(energy_curves)
        tabu_curves = [tabu_curves[i] for i in sort_idx]
        energy_curves_list = [energy_curves[i] for i in sort_idx]
        # Eliminate duplicate curves, using the information of the energy_curves
        # (if possible) to accelerate this process
        duplicates_idx = []
        # The current curves should not be duplicated, we check with the 
        # tabu curves if they are duplicated.
        for curve1 in current_curves:
            for idx, curve2 in enumerate(tabu_curves):
                if (curve1-curve2).H1_norm() < config.H1_tolerance:
                    duplicates_idx.append(idx)
        ## eliminate duplicated idx's and sort them
        duplicates_idx = list(dict.fromkeys(duplicates_idx))
        duplicates_idx.sort(reverse=True)
        # remove the duplicate tabu curves
        for i in duplicates_idx:
            tabu_curves.pop(i)
            energy_curves_list.pop(i)
        print("Eliminating duplicas, eliminated {} duplicate tabu curves".format(
                                                        len(duplicates_idx)))
        # Tabu curves should not be replicated given the Tabu search algorith.
        # Now trim if the curves_list is too long
        pop_counter = 0
        while len(tabu_curves) + N_current_curves > config.curves_list_length_lim and \
              energy_curves_list[-1] >= -1:
            # find the one with least energy and pop it
            tabu_curves.pop()
            energy_curves_list.pop()
            pop_counter += 1
        if pop_counter > 0:
            print("Trimming process: {} low energy tabu curves eliminated".format(
                pop_counter))
        curves_list = current_curves + tabu_curves # joining two lists
    return curves_list, H1_tol_factor

def solve_quadratic_program(current_measure, energy_curves = None,
                                      H1_tol_factor = 1):
    assert isinstance(current_measure, curves.measure)
    # Build the quadratic system of step 5 and then use some generic python
    # solver to get a solution.
    # Build matrix Q and vector b
    logger = config.logger
    logger.status([1,2,1])
    # First, check that no curves are duplicated
    curves_list, H1_tol_factor = \
            measure_trimming(current_measure, energy_curves = energy_curves,
                             H1_tol_factor = H1_tol_factor)
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
    # Here we force Q to be positive semi-definite for the qpsolvers to work
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
#        coefficients = qpsolvers.solve_qp(Q, 1-b, -np.eye(N), np.zeros(N), np.zeros((N,N)), np.zeros(N) )
        Qc = cvxopt.matrix(QQ)
        bb = cvxopt.matrix(1 - b.reshape(-1,1))
        G  = cvxopt.matrix(-np.eye(N))
        h  = cvxopt.matrix(np.zeros((N,1)))
        sol = cvxopt.solvers.qp(Qc,bb,G,h)
        coefficients = np.array(sol['x']).reshape(-1)
    except Exception as e:
        print(e)
        print("Failed to use qpsolvers, try to make it work")
        import dill # to pickle the whole interpreter session
        dill.dump_session('crashdump.pkl')
        # If you want to load later
        # dill.load_session('crashdump.pkl')
        import code; code.interact(local=dict(globals(), **locals()))
    # Incorporate as 0 coefficients those of the duplicates
    coefficients = list(coefficients)
    logger.status([1,2,2], coefficients)
    return curves_list, coefficients

def coefficient_optimization_step(current_measure, energy_curves = None):
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

def gradient_flow_and_optimize(current_measure):
    assert isinstance(current_measure, curves.measure)
    # Method that for a given measure, applies gradient flow on the current
    # curves to shift them, seeking to minimize the main problem's energy.
    # This method intercalates gradient flow methods and optimization steps.
    # Input and output: measure type object.
    stepsize = config.g_flow_init_step
    total_iterations = 0
    while total_iterations <= config.g_flow_opt_max_iter:
        current_measure, stepsize, iters = gradient_flow(current_measure,
                                                               stepsize)
        total_iterations += config.g_flow_opt_in_between_iters
        current_measure = coefficient_optimization_step(current_measure)
        if stepsize < config.g_flow_limit_stepsize:
            # The gradient flow converged, but since the coefficients got 
            # optimized, it is required to restart the gradient flow.
            stepsize = np.sqrt(config.g_flow_limit_stepsize)
        if iters == 0:
            # The current measure is already optimal, therefore, there is no
            # need to keep iterating
            break
    return current_measure

def gradient_flow(current_measure, init_step,
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

def dual_gap(current_measure, curve_list, energies):
    assert isinstance(current_measure, curves.measure) \
            and isinstance(curve_list, list) and isinstance(energies, list)
    # It computes the dual gap between the current measure and the family 
    # of obtained curves via Tabu search. The idea is to just pick the least
    # step3 energy one. 
    arg_min = np.argmin(energies)
    min_curve = curve_list[arg_min]
    compare_measure = curves.measure()
    compare_measure.add(min_curve,1)
    w_t = op.w_t(current_measure)
    M_0 = op.int_time_H_t_product(config.f_t, config.f_t)/2
    c_0 = M_0*compare_measure.integrate_against(w_t)
    compare_measure.modify_intensity(0, c_0)
    val_1 = sum(current_measure.intensities)
    val_2 = op.overpenalization(c_0, M_0)
    val_3 = current_measure.integrate_against(w_t) - \
                compare_measure.integrate_against(w_t)
    return val_1 - val_2 - val_3, c_0

def dual_gap2(current_measure, tabu_curves):
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
        tabu_curves (list of curves class):
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
    insertion_step_minimizer = tabu_curves[0]
    # Build a measure with the global minimizer
    compare_measure = curves.measure()
    compare_measure.add(insertion_step_minimizer, 1)
    # Formula
    return M_0*(compare_measure.integrate_against(w_t)**2 - 1)/2


