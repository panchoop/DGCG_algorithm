import curves
import operators
import optimization as opt
import config
import numpy as np
import misc
import os

""" General controller of the whole DGCG algorithm."""

def set_parameters(time_samples, H_dimensions, test_func, grad_test_func,
                   **kwargs):
    """Method to set the parameters that define the required spaces.

    Before setting any forward operator or indicate the known data, the
    solver requires to know the time samples of the problem, together with
    the dimensions of the considered complex spaces that define the domain.
    Basically, we require:
        t_0 < t_1 < t_2 < ... < t_T ∈ R, the fixed time samples.

        n_0,  n_1,  n_2, ....   n_T ∈ N, the dimensions of each H_{t_i},
        which in this implementation, correspond to the complex space C^{n_i}
        with realified inner product:
            for f,g ∈ C^{n_i}, the product <f,g>_{H_{n_i}} is
                        Re(np.dot(f,np.conf(g)))/n_i

    After setting these parameters, it is possible to use the curve and measure
    classes.

    This method is also required to activate the logging method that will
    record and save each of the steps of the algorithm.
    ---------------------
    Inputs:
        time_samples (list of np.float):
            ordered list of numbers that represent each time sample.
        H_dimensions (list of np.int):
            list of integer numbers representing each H_t dimension.
    Outputs:
        None
    Keyworded arguments:
        None
    """
    default_parameters = {
        "use_ffmpeg": True,
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
    if params['use_ffmpeg'] == False:
        print("WARNING: ffmpeg disabled. Videos of animations cannot be saved")
        misc.use_ffmpeg = params['use_ffmpeg']

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
    os.system("mkdir {}".format(config.temp_folder))
    # <+TODO+> function that saves the used configuration into the temp folder
    logger = config.logger

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
        current_measure = opt.insertion_step(current_measure)
        if current_measure is None:
           break
        logger.status([2],num_iter, current_measure)
        current_measure = opt.gradient_flow_and_optimize(current_measure)

    print("Finished execution")




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




