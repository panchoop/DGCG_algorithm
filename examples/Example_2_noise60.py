""" Script to run Experiment 2 of the paper with 60 % noise

This experiment consist of 3 curves with non-constant speeds.
"""
# Standard imports
import sys
import os
import itertools
import pickle
import numpy as np

# Import package from sibling folder
sys.path.insert(0, os.path.abspath('..'))
from src import DGCG

# General simulation parameters
ALPHA = 0.1
BETA = 0.1
T = 51
TIME_SAMPLES = np.linspace(0, 1, T)

MAX_FREQUENCY = 15
MAX_ANGLES = 5
ANGLES = np.linspace(0, np.pi, MAX_ANGLES)[:-1]
SPACING = 1


def sample_line(num_samples, angle, spacing):
    """ Obtain equally spaced points on lines crossing the origin."""
    rotation_mat = np.array([[np.cos(angle), np.sin(angle)],
                             [-np.sin(angle), np.cos(angle)]])
    horizontal_samples = [-spacing*(i+1) for i in range(num_samples//2)]
    horizontal_samples.extend([spacing*(i+1)
                               for i in range(num_samples - num_samples//2)])
    horizontal_samples = [np.array([x, 0]) for x in horizontal_samples]
    rot_samples = [samp@rotation_mat for samp in horizontal_samples]
    return rot_samples


def available_frequencies(angle):
    """ Avaiable frequencies at angle.

    At each time we cicle through different angles"""
    av_samps = [np.array([0, 0])]
    av_samps.extend(sample_line(MAX_FREQUENCY-1, angle, SPACING))
    return np.array(av_samps)


angle_cycler = itertools.cycle(ANGLES)
FREQUENCIES = [available_frequencies(next(angle_cycler))
               for t in range(T)]
FREQ_DIMENSION = np.ones(T, dtype=int)*MAX_FREQUENCY

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

    # Generate data. The curves with different velocities and trajectories.
    # For this we use the DGCG.classes.curve and DGCG.classes.measure classes.

    # middle curve: straight curve with constant speed.
    start_pos_1 = [0.1, 0.1]
    end_pos_1 = [0.7, 0.8]
    curve_1 = DGCG.classes.curve(np.linspace(0, 1, 2),
                                 np.array([start_pos_1, end_pos_1]))
    # top curve: a circle segment
    times_2 = np.linspace(0, 1, T)
    center = np.array([0.1, 0.9])
    radius = np.linalg.norm(center-np.array([0.5, 0.5]))-0.1
    x_2 = radius*np.cos(3*np.pi/2 + times_2*np.pi/2) + center[0]
    y_2 = radius*np.sin(3*np.pi/2 + times_2*np.pi/2) + center[1]
    positions_2 = np.array([x_2, y_2]).T
    curve_2 = DGCG.classes.curve(times_2, positions_2)
    # bottom curve: circular segment + straight segment and non-constant speed.
    tangent_time = 0.5
    tangent_point = curve_1.eval(tangent_time)
    tangent_direction = curve_1.eval(tangent_time)-curve_1.eval(0)
    normal_direction = np.array([tangent_direction[0, 1],
                                 -tangent_direction[0, 0]])
    normal_direction = normal_direction/np.linalg.norm(tangent_direction)
    radius = 0.3
    init_angle = 4.5*np.pi/4
    end_angle = np.pi*6/16/2
    diff_angle = init_angle - end_angle
    middle_time = 0.8
    center_circle = tangent_point + radius*normal_direction
    increase_factor = 1
    increase = lambda t: increase_factor*t**2 + (1-increase_factor)*t
    times_3 = np.arange(0, middle_time, 0.01)
    times_3 = np.append(times_3, middle_time)
    x_3 = np.cos(init_angle - increase(times_3/middle_time)*diff_angle)
    x_3 = radius*x_3 + center_circle[0, 0]
    y_3 = np.sin(init_angle - increase(times_3/middle_time)*diff_angle)
    y_3 = radius*y_3 + center_circle[0, 1]
    # # # straight line
    times_3 = np.append(times_3, 1)
    middle_position = np.array([x_3[-1], y_3[-1]])
    last_speed = 1
    last_position = middle_position*(1-last_speed) + last_speed*center_circle
    x_3 = np.append(x_3, last_position[0, 0])
    y_3 = np.append(y_3, last_position[0, 1])
    positions_3 = np.array([x_3, y_3]).T
    curve_3 = DGCG.classes.curve(times_3, positions_3)

    # Include these curves inside a measure, with respective intensities
    measure = DGCG.classes.measure()
    intensity_1 = 1
    weight_1 = intensity_1*curve_1.energy()
    measure.add(curve_1, weight_1)
    intensity_2 = 1
    weight_2 = intensity_2*curve_2.energy()
    measure.add(curve_2, weight_2)
    intensity_3 = 1
    weight_3 = intensity_3*curve_3.energy()
    measure.add(curve_3, weight_3)
    # uncomment the next line see the animated curve
    # measure.animate()

    # Simulate the measurements generated by this curve
    data = DGCG.operators.K_t_star_full(measure)
    # uncomment the next line to see the backprojected data
    # dual_variable = DGCG.classes.dual_variable(DGCG.classes.measure())
    # dual_variable._data = -data
    # ani_1 = dual_variable.animate(measure = measure, block = True)

    # Add noise to the measurements. The noise vector is saved in ./annex
    noise_level = 0.6
    noise_vector = pickle.load(open('annex/noise_vector.pickle', 'rb'))
    nois_norm = DGCG.operators.int_time_H_t_product(noise_vector, noise_vector)
    noise_vector = noise_vector/np.sqrt(nois_norm)
    data_H_norm = np.sqrt(DGCG.operators.int_time_H_t_product(data, data))
    data_noise = data + noise_vector*noise_level*data_H_norm

    # uncomment to see the noisy backprojected data
    # dual_variable = DGCG.classes.dual_variable(DGCG.classes.measure())
    # dual_variable._data = -data
    # ani_2 = dual_variable.animate(measure = measure, block = True)

    # settings to speed up the convergence.
    simulation_parameters = {
        'insertion_max_restarts': 10000,
        'results_folder': 'results_Example_2_noise60',
        'multistart_pooling_num': 1000,
        'TOL': 10**(-10)
    }
    # Compute the solution
    solution_measure = DGCG.solve(data_noise, **simulation_parameters)

