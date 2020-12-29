"""
Test to ensure that any modification compiles and outputs the same expected
soltion. The test is expected to run fast therefore we use low number of
frequencies and low number of time samples.
"""
# Standard imports
import sys
import os
import shutil
import pickle
import numpy as np

# Import package from sibling folder
sys.path.insert(0, os.path.abspath('..'))
from src import DGCG

def template(alpha, beta, num_times, h_dim, tol, test_num):
    # General simulation parameters
    alpha = 0.01
    beta = 0.01
    num_times = 5
    h_dim = 5
    TIME_SAMPLES = np.linspace(0, 1, num_times)
    FREQ_DIMENSION = np.ones(num_times, dtype=int)*h_dim

    def archimedian_spiral(time, center_point, loop_dist):
        """ Archimedian spiral to get the frequency measurements"""
        radius = center_point + loop_dist*time
        return np.array([radius*np.cos(time), radius*np.sin(time)])


    FREQ_SAMPLES = np.array([archimedian_spiral(t, 0, 0.2)
                             for t in np.arange(FREQ_DIMENSION[0])])
    FREQUENCIES = [FREQ_SAMPLES for t in range(num_times)]  # at each time sample

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


    # Input of parameters into model.
    # This has to be done first since these values fix the set of extremal
    # points and their generated measurement data
    DGCG.set_model_parameters(alpha, beta, TIME_SAMPLES, FREQ_DIMENSION,
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
    # dual_variable = DGCG.operators.w_t(DGCG.classes.measure())
    # dual_variable.data = -data
    # ani_1 = dual_variable.animate(measure = measure, block = True)

    # (Optionally) Add noise to the measurements
    noise_level = 0
    noise_vector = np.random.randn(*np.shape(data))
    data_H_norm = np.sqrt(DGCG.operators.int_time_H_t_product(data, data))
    data_noise = data + noise_vector*noise_level*data_H_norm

    # uncomment to see the noisy backprojected data
    # dual_variable = DGCG.operators.w_t(DGCG.classes.measure())
    # dual_variable.data = -data
    # ani_2 = dual_variable.animate(measure = measure, block = True)

    # non-standard settings: fixing the configurations in config.py
    DGCG.config.multistart_inter_iteration_checkup = 10
    DGCG.config.measure_coefficient_too_low = 1e-18
    DGCG.config.multistart_descent_limit_stepsize = 1e-40
    DGCG.config.H1_tolerance = 1e-5
    DGCG.config.CVXOPT_TOL = 1e-25
    DGCG.config.g_flow_limit_stepsize = 1e-40
    # settings to speed up the convergence.
    simulation_parameters = {
        'insertion_max_restarts': 20,
        'insertion_min_restarts': 5,
        'results_folder': 'results',
        'multistart_pooling_num': 10,
        'TOL': tol,
    }
    solution_measure = DGCG.solve(data_noise, **simulation_parameters)[0]
    shutil.rmtree(simulation_parameters['results_folder'])

    solution_filename = 'compare_results/test_{}_sol.pickle'.format(test_num)
    # dump solution
    # pickle.dump(solution_measure, open(solution_filename, 'wb'))

    # compare solution
    saved_solution = pickle.load(open(solution_filename, 'rb'))

    # compare number of atoms
    sol_atom_number = len(solution_measure.curves)
    saved_atom_number = len(saved_solution.curves)
    diff_atom_number = abs(sol_atom_number - saved_atom_number)
    print('Difference of number of atoms: {}'.format(diff_atom_number))

    # compare energies
    sol_energy = solution_measure.get_main_energy()
    saved_energy = saved_solution.get_main_energy()
    diff_energy = abs(sol_energy - saved_energy)
    print('Difference of main energies is: {}'.format(diff_energy))
    assert diff_energy < tol, "Failed test"


def test_1():
    alpha = 0.01
    beta = 0.01
    tol = 10**-10
    num_times = 5
    h_dim = 5
    test_num = '01'
    template(alpha, beta, num_times, h_dim, tol, test_num)


def test_2():
    alpha = 0.1
    beta = 0.01
    tol = 10**-10
    num_times = 5
    h_dim = 5
    test_num = '02'
    template(alpha, beta, num_times, h_dim, tol, test_num)


def test_3():
    alpha = 0.01
    beta = 0.1
    tol = 10**-10
    num_times = 5
    h_dim = 5
    test_num = '03'
    template(alpha, beta, num_times, h_dim, tol, test_num)


def test_4():
    alpha = 0.01
    beta = 0.01
    tol = 10**-13
    num_times = 5
    h_dim = 5
    test_num = '04'
    template(alpha, beta, num_times, h_dim, tol, test_num)


def test_5():
    alpha = 0.01
    beta = 0.01
    tol = 10**-10
    num_times = 10
    h_dim = 5
    test_num = '05'
    template(alpha, beta, num_times, h_dim, tol, test_num)


def test_6():
    alpha = 0.01
    beta = 0.01
    tol = 10**-10
    num_times = 5
    h_dim = 10
    test_num = '06'
    template(alpha, beta, num_times, h_dim, tol, test_num)


test_4()

