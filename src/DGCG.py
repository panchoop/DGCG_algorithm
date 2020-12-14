""" General controller of the whole DGCG algorithm."""
# Standard imports
import numpy as np
import os

# Local imports
from . import curves, operators, config, misc, insertion_step, optimization
from . import checker


def set_model_parameters(alpha, beta, time_samples, H_dimensions,
                         test_func, grad_test_func):
    """Method to the the model parameters that define our problem.

    Parameters
    ----------
    alpha, beta: positive double
        The regularization parameters of the regularization problem.
    time_samples: numpy.ndarray
        Ordered array of values between 0 and 1, with time_samples[0] = 0 
        and time_samples[1] = 1.
    H_dimension: List[int]
        List of dimensions of considered Hilbert spaces.
    test_func : function handle
        The function that define the forward operator and its dual.
        These must be a two input function
        φ:[0,1,...,T-1]xΩ, where the first variable corresponds to time
        and the second variable corresponds to spatial point evaluations.
        This function must map into H_t and must tolerate multiple spatial
        point evaluation.
        Parameters:
            t (int from 0 to T-1) the index of a time sample.
            x (numpy.ndarray of size (N,2)) representing N spatial points in
                the considered two-dimensional domain.
        Returns:
            numpy.ndarray of size (N,K), a collection of points in the Hilbert
            space H_T. K is given by H_dimensions[t]
    grad_test_func : function handle
        The gradient of the input function test_func. The inputs of the
        gradient are the same of those of the original function.
        Returns:
            (2xNxK numpy array) representing two collections of N points in
            H_t. Each collection correspond to the partial derivative
            ∂_x and ∂_y respectively.
    Notes
    -----
    It is required to set this values prior to defining atoms or taking
    measurements. This is because the input values fix the set of extremal
    points of the Benomou-Brenier energy, and the given kernels define the
    Forward and Backward measurement operators.
    """
    config.alpha = alpha
    config.beta = beta
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
        raise Exception("the input test function input/output dimensions are" +
                        " not supported; please refer to instructions. " +
                        "execution aborted.")

    if not test_grad_func_check(grad_test_func):
        raise Exception("the input test function gradient input/output " +
                        "dimensions are  not supported; please refer to " +
                        "instructions. execution aborted.")
    operators.test_func = test_func
    operators.grad_test_func = grad_test_func

def solve(data, **kwargs):
    """Method to solve the given dynamic inverse problem for input data.

    This function will apply the Dynamic Generalized Conditional Gradient
    (DGCG) algorithm.

    Parameters
    ----------
    data : numpy.ndarray
        Array with shape (T,K), where T is the number of time samples
        and K is the number of dimensions of the data space.
        T corresponds to len(time_samples), the input of the
        DGCG.set_model_parameters function, and K is the dimensions of the
        considered Hilbert space (given by H_dimensions[0], here we have only
        implemented the constant dimension case).
    initial_measure : DGCG.curves.measure class, optional
        Initial guess for the DGCG algorithm. Default value is `None`
        corresponding the the zero measure.
    use_ffmmpeg : Boolean, optional
        To indicate the use of the `ffmpeg` library. If set to false,
        matplotlib won't be able to save the output videos as videos files.
        Nonetheless, it is possible to animate the measures with the
        `DGCG.curves.measure.animate` method.
    insertion_max_restarts : int, optional
        Hard limit on the number of allowed restarts for the multistart
        gradient descent at each iteration. Default 1000.
    insertion_min_restarts : int, optional
        Hard limit on the number of allowed restarts for the multistart
        gradient descent at each iteration. Default 20.
    results_folder : str, optional
        name of the folder that will be created to save the simulation
        results. Default 'results'.
    multistart_early_stop : function, optional
        function to stop early as a function of the found stationary points.
        Default lambda n: np.inf. The default setting iterates until reaching
        the insertion_max_restarts value.
    multistart_pooling_num : int, optional
        When insertion random curves, the algorithm will realize this given
        number of curves and then choose the one with best F(γ) value.
        The higher the value of this parameter, the more one
        samples on the best initial curves to descent. The drawback
        is that it slows down the proposition of random curves.
    log_output : Boolean, optional
        Save the output of shell into a .txt inside the results folder.
        default False, to be improved. <+TODO+>

    Returns
    -------
    solution (DGCG.curves.measure class)
        Measure clsas object that represents the obtained solution.
    exit_flat ( (int, str) tuple)
        Tuple with a numeric indicator and a string with a brief description.
        <+TODO+> check this, add dual_gap exit value.
    keyworded arguments: None
    """
    default_parameters = {
        'initial_measure': None,
        "use_ffmpeg": True,
        "insertion_max_restarts": 1000,
        "insertion_min_restarts": 20,
        "results_folder": "results",
        "multistart_early_stop": lambda n: np.inf,
        "multistart_pooling_num": 1000,
        "log_output": False,
        "insertion_max_segments": 5,
    }
    for key, val in kwargs.items():
        if key not in default_parameters.keys():
            raise Exception("The input keyworded argument do no exists")
        default_parameters[key] = val
    params = default_parameters
    # Set the parameters
    if not params['use_ffmpeg']:
        print("WARNING: ffmpeg disabled. Videos of animations cannot be saved")
        misc.use_ffmpeg = params['use_ffmpeg']
    else:
        misc.use_ffmpeg = True
    if params['insertion_max_restarts'] < params['insertion_min_restarts']:
        raise Exception("insertion_max_restarts < insertion_min_restarts." +
                        "The execution is aborted")
    config.insertion_max_restarts = params['insertion_max_restarts']
    config.insertion_min_restarts = params['insertion_min_restarts']
    config.results_folder = params['results_folder']
    config.multistart_early_stop = params['multistart_early_stop']
    config.multistart_pooling_num = params['multistart_pooling_num']
    if params['log_output']:
        config.log_output = params['log_output']
        config.logger.logtext = ''
        config.logger.logcounter = 0
    config.insertion_max_segments = params['insertion_max_segments']

    # Input the parameters into their respective modules
    if not checker.is_in_H(data):
        raise Exception("The input data has the wrong dimensions.")
    config.f_t = data

    # Folder to store the simulation results
    os.makedirs(config.results_folder)

    # Logger class, storing the used parameters
    logger = misc.logger()
    config.logger = logger
    logger.log_config('{}/config.pickle'.format(config.results_folder))

    # Initial guess definition: the zero measure by default
    if default_parameters['initial_measure'] is None:
        current_measure = curves.measure()
        M_0 = operators.int_time_H_t_product(data, data)/2
        current_measure.main_energy = M_0
    else:
        if isinstance(default_parameters['initial_measure'], curves.measure):
            current_measure = default_parameters['initial_measure']
            _ = current_measure.get_main_energy()

    for num_iter in range(1, config.full_max_iterations):
        logger.status([1], num_iter, current_measure)
        current_measure, flag = insertion_step.insertion_step(current_measure)
        logger.status([2], num_iter, current_measure)
        if flag == 0:
            print("Finished execution")
            return current_measure, (1, 'SUCCESS')
        current_measure = optimization.slide_and_optimize(current_measure)
    print("Maximum number of iterations ({}) reached!".format(
                                                config.full_max_iterations))
    return current_measure, (0, 'FAILURE: unable to reach a solution')




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



