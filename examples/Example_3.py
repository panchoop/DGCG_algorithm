""" Script to run Experiement 3 of the paper

This experiment consists of two crossing curves with constant velocity and
equal intensity.
"""
# Standard imports
import sys
import os
import numpy as np

# Import package from sibling folder
sys.path.insert(0, os.path.abspath('..'))
from src import DGCG

# General simulation parameters
T = 51
TIME_SAMPLES = np.linspace(0, 1, T)
FREQ_DIMENSION = np.ones(T, dtype=int)*18

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
def test_func(t, x):  # φ_t(x)
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

def grad_test_func(t, x):  # ∇φ_t(x)
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
    # time, frequency and kernels are input into the module.
    DGCG.set_parameters(TIME_SAMPLES, FREQ_DIMENSION,
                        test_func, grad_test_func)

    # Generate data. Two crossing curves with constant velocity
    # For this we use the DGCG.curves.curve and DGCG.curves.measure classes.

    # First curve, straight one.
    initial_position_1 = [0.2, 0.2]
    final_position_1 = [0.8, 0.8]
    positions_1 = np.array([initial_position_1, final_position_1])
    times_1 = np.array([0, 1])
    curve_1 = DGCG.curves.curve(times_1, positions_1)

    # Second curve, also a straight one with opposite direction
    initial_position_2 = [0.8, 0.2]
    final_position_2 = [0.2, 0.8]
    positions_2 = np.array([initial_position_2, final_position_2])
    times_2 = np.array([0, 1])
    curve_2 = DGCG.curves.curve(times_2, positions_2)

    # Include these curves inside a measure, with respective intensities
    intensity_1 = 1
    intensity_2 = 1
    measure = DGCG.curves.measure()
    measure.add(curve_1, intensity_1)
    measure.add(curve_2, intensity_2)
    # uncomment the next line see the animated curve
    # measure.animate()

    # Simulate the measurements generated by this curve
    data = DGCG.operators.K_t_star_full(measure)
    # uncomment the next line to see the backprojected data
    # dual_variable = DGCG.operators.w_t(DGCG.curves.measure())
    # dual_variable.data = -data
    # ani_1 = dual_variable.animate(measure = measure, block = True)

    # (Optionally) Add noise to the measurements
    noise_level = 0
    noise_vector = np.random.randn(*np.shape(data))
    data_H_norm = np.sqrt(DGCG.operators.int_time_H_t_product(data, data))
    data_noise = data + noise_vector*noise_level*data_H_norm

    # uncomment to see the noisy backprojected data
    # dual_variable = DGCG.operators.w_t(DGCG.curves.measure())
    # dual_variable.data = -data
    # ani_2 = dual_variable.animate(measure = measure, block = True)

    # Additional parameters to input to the DGCG solver
    alpha = 0.2
    beta = 0.2

    parameters = {
        'insertion_max_restarts': 50,
        'insertion_min_restarts': 20,
        'results_folder': 'results_Exercise_3'
    }
    DGCG.set_parameters(TIME_SAMPLES, FREQ_DIMENSION,
                        test_func, grad_test_func, **parameters)

    # Compute the solution
    current_measure = DGCG.solve(data_noise, alpha, beta)
