""" Script to run Experiement 1 of the paper

This experiment consist of a single curve with constant speed.
"""
# Standard imports
import sys
import os
import numpy as np

# Import package from sibling folder
sys.path.insert(0, os.path.abspath('..'))
from src import DGCG

# General simulation parameters
ALPHA = 0.1
BETA = 0.1
T = 51
TIME_SAMPLES = np.linspace(0, 1, T)
FREQ_DIMENSION = np.ones(T, dtype=int)*20

def Archimedian_spiral(t, a, b):
    """ Archimedian spiral to get the frequency measurements"""
    return np.array([(a+b*t)*np.cos(t), (a+b*t)*np.sin(t)])


FREQ_SAMPLES = np.array([Archimedian_spiral(t, 0, 0.2)
                         for t in np.arange(FREQ_DIMENSION[0])])
FREQUENCIES = [FREQ_SAMPLES for t in range(T)]  # at each time sample

# Some helper functions
def cut_off(s):
    """ One-dimensiona twice-differentiable monotonous cut-off function.

    This function is applied to the forward measurements such that the
    boundary of [0,1]x[0,1] is never reached.
    cutoff_threshold define the intervals [0,h], [1-h, h] where the
    cut_off values are neither 0 or 1.

    Parameters
    ----------
    s : numpy.ndarray
        1-dimensional array with values in [0,1]

    Returns
    -------
    numpy.ndarray
        1-dimensional array with values in [0,1]

    """
    cutoff_threshold = 0.1
    transition = lambda s: 10*s**3 - 15*s**4 + 6*s**5
    val = np.zeros(s.shape)
    for idx, s_i in enumerate(s):
        if cutoff_threshold <= s_i <= 1-cutoff_threshold:
            val[idx] = 1
        elif 0 <= s_i < cutoff_threshold:
            val[idx] = transition(s_i/cutoff_threshold)
        elif 1-cutoff_threshold < s_i <= 1:
            val[idx] = transition(((1-s_i))/cutoff_threshold)
        else:
            val[idx] = 0
    return val


def D_cut_off(s):
    """ Derivative of the one-dimensional cut-off functionk

    Parameters
    ----------
    s : numpy.ndarray
        1-dimensional array with values in [0,1]

    Returns
    -------
    numpy.ndarray of dimension 1.
    """
    cutoff_threshold = 0.1
    D_transition = lambda s: 30*s**2 - 60*s**3 + 30*s**4
    val = np.zeros(s.shape)
    for idx, s_i in enumerate(s):
        if 0 <= s_i < cutoff_threshold:
            val[idx] = D_transition(s_i/cutoff_threshold)/cutoff_threshold
        elif 1-cutoff_threshold < s_i <= 1:
            val[idx] = -D_transition((1-s_i)/cutoff_threshold)/cutoff_threshold
        else:
            val[idx] = 0
    return val


# Kernels to define the forward measurements
def TEST_FUNC(t, x):  # φ_t(x)
    """ Kernel that define the forward measurements

    Fourier frequency measurements sampled at the pre-set `FREQUENCIES`
    and cut-off at the boundary of [0,1]x[0,1].

    Parameters
    ----------
    t : int
        The time sample of evaluation, ranges from 0 to T-1
    x : numpy.ndarray of size (N,2)
        represents a list of N 2-dimensional spatial points.

    Returns
    -------
    numpy.ndarray of size (N,K)
        The evaluations of the test function in the set of N spatial
        points. K corresponds to the number of FREQUENCIES at the input
        time.

    Notes
    -----
    The output of this function is an element of the Hilbert space H_t
    as described in the theory.
    """
    input_fourier = x@FREQUENCIES[t].T  # size (N,K)
    fourier_evals = np.exp(-2*np.pi*1j*input_fourier)
    cutoff = cut_off(x[:, 0:1])*cut_off(x[:, 1:2])
    return fourier_evals*cutoff

def GRAD_TEST_FUNC(t, x):  # ∇φ_t(x)
    """ Gradient of kernel that define the forward measurements

    Gradient of Fourier frequency measurements sampled at the pre-set
    `FREQUENCIES` and cut-off at the boundary of [0,1]x[0,1].

    Parameters
    ----------
    t : int
        The time sample of evaluation, ranges from 0 to T-1
    x : numpy.ndarray of size (N,2)
        represents a list of N 2-dimensional spatial points.

    Returns
    -------
    numpy.ndarray of size (2,N,K)
        The evaluations of the gradient of the kernel in the set of N
        spatial points. K corresponds to the number of FREQUENCIES at the
        input time. The first variable references to the gradient on
        the first dimension, and the second dimensions respectively.

    Notes
    -----
    The output of this function are two elements of the Hilbert space H_t
    as described in the theory.
    """
    freq_t = FREQUENCIES[t]
    input_fourier = x@freq_t.T  # size (N,K)
    fourier_evals = np.exp(-2*np.pi*1j*input_fourier)
    # cutoffs
    cutoff_1 = cut_off(x[:, 0:1])
    cutoff_2 = cut_off(x[:, 1:2])
    D_cutoff_1 = D_cut_off(x[:, 0:1])
    D_cutoff_2 = D_cut_off(x[:, 1:2])
    D_constant_1 = -2*np.pi*1j*cutoff_1@freq_t[:, 0:1].T + D_cutoff_1
    D_constant_2 = -2*np.pi*1j*cutoff_2@freq_t[:, 1:2].T + D_cutoff_2
    # wrap it up
    output = np.zeros((2, x.shape[0], FREQ_DIMENSION[t]), dtype='complex')
    output[0] = fourier_evals*cutoff_2*D_constant_1
    output[1] = fourier_evals*cutoff_1*D_constant_2
    return output


if __name__ == "__main__":
    # Input of parameters into model.
    # This has to be done first since these values fix the set of extremal
    # points and their generated measurement data
    DGCG.set_model_parameters(ALPHA, BETA, TIME_SAMPLES, FREQ_DIMENSION,
                              TEST_FUNC, GRAD_TEST_FUNC)

    # Generate data. A single crossing curve with constant velocity
    # For this we use the DGCG.classes.curve and DGCG.classes.measure classes.

    # A straight curve.
    initial_position = [0.2, 0.2]
    final_position = [0.8, 0.8]
    positions = np.array([initial_position, final_position])
    times = np.array([0, 1])
    curve = DGCG.classes.curve(times, positions)

    # Include these curves inside a measure, with respective intensities
    intensity = 1
    weight = intensity*curve.energy()
    measure = DGCG.classes.measure()
    measure.add(curve, weight)
    # uncomment the next line see the animated curve
    # measure.animate()

    # Simulate the measurements generated by this curve
    data = DGCG.operators.K_t_star_full(measure)
    # uncomment the next line to see the backprojected data
    # dual_variable = DGCG.classes.dual_variable(DGCG.classes.measure())
    # dual_variable._data = -data
    # ani_1 = dual_variable.animate(measure = measure, block = True)

    # (Optionally) Add noise to the measurements
    noise_level = 0
    noise_vector = np.random.randn(*np.shape(data))
    data_H_norm = np.sqrt(DGCG.operators.int_time_H_t_product(data, data))
    data_noise = data + noise_vector*noise_level*data_H_norm

    # uncomment to see the noisy backprojected data
    # dual_variable = DGCG.classes.dual_variable(DGCG.classes.measure())
    # dual_variable._data = -data
    # ani_2 = dual_variable.animate(measure = measure, block = True)

    # settings to speed up the convergence.
    simulation_parameters = {
        'insertion_max_restarts': 20,
        'insertion_min_restarts': 5,
        'results_folder': 'results_Exercise_1',
        'multistart_pooling_num': 100,
    }
    # Compute the solution
    solution_measure = DGCG.solve(data_noise, **simulation_parameters)
