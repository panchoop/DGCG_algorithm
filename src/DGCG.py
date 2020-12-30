""" General controller of the DGCG algorithm package.
"""
# Standard imports
import numpy as np
import os

# Local imports
from . import classes, operators, config, misc, insertion_step, optimization
from . import log_mod
from . import checker


def set_model_parameters(alpha, beta, time_samples, H_dimensions,
                         test_func, grad_test_func):
    """Set the the fundamental parameters of the model.

    Parameters
    ----------
    alpha, beta: float
        Regularization parameter of the regularization problem, must be
        positive.
    time_samples: numpy.ndarray
        Ordered array of values between 0 and 1, with ``time_samples[0] = 0``
        and ``time_samples[-1] = 1``.
    H_dimension: list[int]
        List of dimensions of the considered Hilbert spaces ``H_t``.
    test_func : callable[[int, numpy.ndarray], numpy.ndarray]
        Function φ that defines the forward measurements. The first input
        is time, the second input is a list of elements in the domain Ω. It
        maps into a list of elements in H_t. See Notes for further reference.
    grad_test_func : callable[[int, numpy.ndarray], numpy.ndarray]
        The gradient of the input function `test_func`. The inputs of the
        gradient are the same of those of the original function.
        Returns a tuple with each partial derivative.

    Returns
    -------
    None

    Notes
    -----
    It is required to set this values prior to defining atoms or taking
    measurements. This is because the input values fix the set of extremal
    points of the Benomou-Brenier energy, and the given kernels define the
    Forward and Backward measurement operators.

    The ``test_func`` φ is the funciton that defines the forward measurements.
    The first input is a time sample in ``[0, 1, ..., T-1]``, with ``T`` the
    total number of time samples. The second input is a list of ``N`` elements
    in Ω, expressed as a (N,2) ``numpy.ndarray`` (Ω is of dimension 2).

    The output of φ is a list of ``N`` elements in ``H_t``, since the
    dimension of ``H_t`` is input with ``H_dimensions``, then the output
    of φ(t, x) is a (N, H_dimensions[t]) ``numpy.ndarray``

    The function ``grad_test_func`` ∇φ has the same input, but the output is
    a (2, N, H_dimensions[t]) tuple representing the two partial derivatives
    ∂_x and ∂_y respectively.
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
    operators.H_DIMENSIONS = rounded_H_dim
    operators.TEST_FUNC = test_func
    operators.GRAD_TEST_FUNC = grad_test_func


def solve(data, **kwargs):
    """Solve the given dynamic inverse problem for input data.

    This function will apply the Dynamic Generalized Conditional Gradient
    (DGCG) algorithm.

    Parameters
    ----------
    data : numpy.ndarray
        Array of ``T`` entries, each a numpy.ndarray of size ``H_dimensions[t]``
        for each ``t``. See notes for further reference.
    initial_measure : :py:class:`src.classes.measure`, optional
        Initial guess for the DGCG algorithm. Default value is `None`
        corresponding the the zero measure.
    use_ffmmpeg : bool, optional
        To indicate the use of the `ffmpeg` library. If set to false,
        matplotlib won't be able to save the output videos as videos files.
        Nonetheless, it is possible to animate the measures with the
        `DGCG.classes.measure.animate` method.
    insertion_max_restarts : int, optional
        Hard limit on the number of allowed restarts for the multistart
        gradient descent at each iteration. Default 1000.
    insertion_min_restarts : int, optional
        Hard limit on the number of allowed restarts for the multistart
        gradient descent at each iteration. Default 20.
    results_folder : str, optional
        name of the folder that will be created to save the simulation
        results. Default 'results'.
    multistart_early_stop : callable[[int,int], int] optional
        function to stop early as a function of the found stationary points.
        Default lambda n,m: np.inf.
    multistart_pooling_num : int, optional
        When insertion random curves, the algorithm will realize this given
        number of curves and then choose the one with best F(γ) value.
        The higher the value of this parameter, the more one
        samples on the best initial curves to descent. The drawback
        is that it slows down the proposition of random curves.
    log_output : bool, optional
        Save the output of shell into a .txt inside the results folder.
        default False, to be improved. <+TODO+>

    Returns
    -------
    solution : :py:class:`src.classes.measure`
        The computed solution.
    exit_flag : tuple[int, str]
        Tuple with a numeric indicator and a string with a brief description.
        <+TODO+> check this, add dual_gap exit value.

    Notes
    -----
    The ``data`` input corresponds to the gathered data with the defined
    forward operator when running :py:func:`src.DGCG.set_model_parameters`.
    Each entry of this array correspond to the measurement at each time sample.
    Therefore, the size of that entry will correspond to the respective ``H_t``
    space.
    """
    default_parameters = {
        'initial_measure': None,
        "use_ffmpeg": True,
        "insertion_max_restarts": 1000,
        "insertion_min_restarts": 20,
        "results_folder": "results",
        "multistart_early_stop": lambda num_tries, num_found: np.inf,
        "multistart_pooling_num": 1000,
        "log_output": False,
        "insertion_max_segments": 5,
        "TOL": 10**(-10)
    }
    for key, val in kwargs.items():
        if key not in default_parameters.keys():
            raise Exception("The input keyworded argument do no exists")
        default_parameters[key] = val
    params = default_parameters
    # Set the parameters
    if not params['use_ffmpeg']:
        print("WARNING: ffmpeg disabled. Videos of animations cannot be saved")
        config.use_ffmpeg = params['use_ffmpeg']
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
    config.insertion_eps = params['TOL']

    # Input the parameters into their respective modules
    if not checker.is_in_H(data):
        raise Exception("The input data has the wrong dimensions.")
    config.f_t = data

    # Folder to store the simulation results
    os.makedirs(config.results_folder)

    # Logger class, storing the used parameters
    logger = log_mod.logger()
    config.logger = logger
    logger.log_config('{}/config.pickle'.format(config.results_folder))

    # Initial guess definition: the zero measure by default
    if default_parameters['initial_measure'] is None:
        current_measure = classes.measure()
        M_0 = operators.int_time_H_t_product(data, data)/2
        current_measure.main_energy = M_0
    else:
        if isinstance(default_parameters['initial_measure'], classes.measure):
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


if __name__ == ' __main__':
    pass



