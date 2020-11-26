# Standard imports
import numpy as np
import os

# Local imports
from . import curves, operators, config, misc, insertion_step
from . import optimization as opt

""" General controller of the whole DGCG algorithm."""

def set_parameters(time_samples, H_dimensions, test_func, grad_test_func,
                   **kwargs):
    """Method to set the parameters that define the required spaces.

    Before setting any forward operator or indicate the known data, the
    solver requires to know the time samples of the problem, together with
    the dimensions of the considered complex spaces that define the domain.

    Basically, we require:
        * t_0 < t_1 < t_2 < ... < t_T ∈ R, the fixed time samples.
        * n_0,  n_1,  n_2, ....   n_T ∈ N, the dimensions of each H_{t_i},
    Which in this implementation, correspond to the complex space C^{n_i} with realified inner product:
        for f,g ∈ C^{n_i}, the product <f,g>_{H_{n_i}} is Re(np.dot(f,np.conf(g)))/n_i

    After setting these parameters, it is possible to use the curve and measure
    classes.

    This method is also required to activate the logging method that will
    record and save each of the steps of the algorithm.

    Args:
        time_sample (list of np.float):
            ordered list of numbers that represent each time sample.
        H_dimensions (list of np.int):
            list of integer numbers representing each H_t dimension.
        test_func (function φ):
            The functions that define the forward operator and its dual.
            These must be a two input function
            φ:[0,1,...,T-1]xΩ, where the first variable corresponds to time
            and the second variable corresponds to spatial point evaluations.
            This function must map into H_t and must tolerate multiple spatial
            point evaluation.
            Inputs:
                t (int from 0 to T-1) representing a specific time sample.
                x (Nx2 numpy array) representing N spatial points in Ω, that
                    is two dimensional.
            Output:
                (NxK numpy array) representing a collection of N points in H_t,
                where K stands for the dimensions of H_t.
        grad_test_func (function ∇φ):
            The gradient of the input function φ. The inputs of the gradient
            are the same of those of the original function.
            Output:
                (2xNxK numpy array) representing two collections of N points in
                H_t. Each collection correspond to the partial derivative
                ∂_x and ∂_y respectively.
    Outputs:
        None
    Keyworded arguments:
        use_ffmmpeg (Boolean, default True):
            To indicate the use of the `ffmpeg` library. If set to false,
            matplotlib won't be able to save the output videos as videos files.
            Nonetheless, it is possible to animate the measures with the
            `.animate` method.
        insertion_max_restarts (integer, default 10000):
            Hard limit on the number of allowed restarts for the multistart
            gradient descent at each iteration.
        insertion_min_restarts (integer, default 15):
            Hard limit on the number of allowed restarts for the multistart
            gradient descent at each iteration.
        results_folder (string, default 'results'):
            name of the folder that will be created to save the simulation
            results
        multistart_early_stop (function, default n*log(n)):
            function to stop early as a function of the found stationary points.
            The default one is log(0.01)/log((n-1)/n).
            This default function is derived by computing the probability
            of missing a particular stationary curve, by assuming that «n»
            is the total number of them and that we choose them by sampling
            randomly on the set of stationary curves. In this case, the number
            of restarts will correspond to having a 1% probability of missing
            the global optimal curve.
        multistart_pooling_num (integer, default 100):
            When insertion random curves, it is possible to generate a batch
            of them and then the just descent from the one with best energy
            value F(γ). The higher the value of this parameter, the more one
            samples on the top tier of initial curves to descent. The drawback
            is that it slows down the proposition of random curves.
        crossover_child_F_threshold (double, default 0.8):
            When crossing over and generating child curves to descent, the
            algorithm will only descend those curves whose F(γ) value is
            below F(γ_max)*crossover_child_F_threshold, with
            F(γ_best) the smallest known F(γ) value among the known stationary
            curves.
    """
    default_parameters = {
        "use_ffmpeg": True,
        "insertion_max_restarts": 10000,
        "insertion_min_restarts": 15,
        "results_folder": "results",
        "multistart_early_stop": lambda n: np.log(0.01)/np.log((n-1)/n),
        "multistart_pooling_num": 100,
        "crossover_child_F_threshold": 0.8,
        "log_output": False,
        "insertion_max_segments":5,
    }
    # Incorporate the input keyworded values
    for key, val in kwargs.items():
        if key in default_parameters:
            default_parameters[key] = val
        else:
            raise KeyError(
               'The given keyworded argument «{}» is not valid'.format(key))
    params = default_parameters
    #
    time_samples = np.array(time_samples)
    # check that it is an ordered array of different elements
    if np.all(np.diff(time_samples) <= 0):
        raise Exception('the given time samples are not ordered or different')
    T = len(time_samples)
    config.T = T
    config.time = time_samples
    # Check that the input dimensions are integers
    tol_error = 1e-10
    rounded_H_dim = np.array([np.round(dim) for dim in H_dimensions], dtype=int)
    if np.any(np.abs(rounded_H_dim - np.array(H_dimensions)) > tol_error):
        raise Exception('The given dimensions are not integer numbers')
    operators.K = rounded_H_dim
    # <+TODO+> Check that the implemented functions map correctly
    if not test_func_check(test_func):
        raise exception("the input test function input/output dimensions are "+
                        " not supported; please refer to instructions. " +
                        "execution aborted.")

    if not test_grad_func_check(grad_test_func):
        raise exception("the input test function gradient input/output "+
                        "dimensions are  not supported; please refer to " +
                        "instructions. execution aborted.")
    operators.test_func = test_func
    operators.grad_test_func = grad_test_func


    # Initialize logger class
    logger = misc.logger()
    config.logger = logger
    # Input additional parameters
    ## use_ffmpeg
    if params['use_ffmpeg'] == False:
        print("WARNING: ffmpeg disabled. Videos of animations cannot be saved")
        misc.use_ffmpeg = params['use_ffmpeg']
    ## insertion_max_restarts
    ## insertion_min_restarts
    if params['insertion_max_restarts'] < params['insertion_min_restarts']:
        print("WARNING: insertion_max_restarts < insertion_min_restarts")
        text = "The value ({}) of insertion_max_restarts got increased to "+ \
              "the same of of the insertion_min_restart ({})"
        print(text.format(params['insertion_max_restarts'],
                          params['insertion_min_restarts']))
        params['insertion_max_restarts'] = params['insertion_min_restarts']
    config.insertion_max_restarts = params['insertion_max_restarts']
    config.insertion_min_restarts = params['insertion_min_restarts']
    ## results_folder
    config.results_folder = params['results_folder']
    ## multistart_early_stop
    config.multistart_early_stop = params['multistart_early_stop']
    ## multistart_pooling_num
    config.multistart_pooling_num = params['multistart_pooling_num']
    ## crossover_child_F_threshold
    config.crossover_child_F_threshold = params['crossover_child_F_threshold']
    ## log_output
    if params['log_output']:
        config.log_output = params['log_output']
        config.logger.logtext = ''
        config.logger.logcounter = 0
    ## insertion_max_segments
    config.insertion_max_segments = params['insertion_max_segments']

def solve( data, alpha, beta, **kwargs):
    """Method to solve the given dynamic inverse problem.

    This function will make some basic checks on the given inputs and then will
    apply the Dynamic Generalized Conditional Gradient (DGCG) algorithm.
    ----------------
    inputs: <+TODO+>

    output:
        solution (measure class object):
            Measure class object that representing the obtained solution.
        exit_flag ( (int, str) tuple):
            Tuple with a numeric indicator and a string with a brief description.

    keyworded arguments: None
    """
    default_parameters = {
        'initial_measure': None,
    }
    for key, val in kwargs.items():
        if not key in default_parameters.keys():
            raise Exception("The input keyworded argument do no exists")
        else:
            default_parameters[key] = val

    # Input the parameters into their respective modules
    config.f_t = data
    config.alpha = alpha
    config.beta = beta

    # Folder to store the simulation results
    os.system("mkdir {}".format(config.results_folder))
    # <+TODO+> function that saves the used configuration into the temp folder
    logger = config.logger
    # log the configuration data of this simulation
    logger.log_config('{}/config.pickle'.config.results_folder)

    # Initial guess definition: the zero measure by default
    if default_parameters['initial_measure'] is None:
        current_measure = curves.measure()
        M_0 = operators.int_time_H_t_product(data, data)/2
        current_measure.main_energy = M_0
    else:
        if isinstance(default_parameters['initial_measure'], curves.measure):
            _ = current_measure.get_main_energy()

    for num_iter in range(1,config.full_max_iterations):
        logger.status([1],num_iter, current_measure)
        current_measure, flag = insertion_step.insertion_step(current_measure)
        if flag == 0:
            print("Finished execution")
            return current_measure
        logger.status([2],num_iter, current_measure)
        current_measure = opt.gradient_flow_and_optimize(current_measure)
    print("Maximum number of iterations ({}) reached!".format(
                                                config.full_max_iterations))




def test_func_check(func):
    """Tests if the dimensions of the given test function φ fit the model.
    """
    # <+to_implement+>
    return True

def test_grad_func_check(grad_func):
    """Tests if the dimensions of the given test function gradient ∇φ fit.
    """
    # <+to_implement+>
    return True

if __name__=='__main__':
    pass



