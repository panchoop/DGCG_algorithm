"""Methods related to the problem's forward operator and Hilbert spaces.

The Hilbert spaces are implicitly defined via the functons in this module.
A priory, these are numpy.ndarray of type complex objects, representing a
finite dimensional complex space.

This module has certain global variables to be used by the defined methods.
These are set by :py:meth:`src.DGCG.set_model_parameters`

Global variables
----------------
test_func : callable
    Function representing the kernel that defines the forward operator.
grad_test_func : callable
    Derivative function of test_func
H_dimensions : list[int]
    Dimensions of each of the considered Hilbert spaces.
"""
import numpy as np
import pyopencl as cl
import pyopencl.array as clarray

# Local imports
from . import config, checker, opencl_mod
from . import classes

TEST_FUNC = None
GRAD_TEST_FUNC = None
H_DIMENSIONS = None

def H_t_product_cl(f_alloc, g_alloc):
    """ Computes the Hilbert space product between two elements in H_t

    Parameters
    ----------
    f_alloc, g_alloc: opencl_mod.mem_alloc
        representing a set of elements in any H_t space. Each of these 
        mem_allocs have size 2, representing the real and imaginary part,
        have dtype np.float64 and shape[-1] = K.
    Returns
    -------
        pyopencl.array of shape (1,)
    """
    assert isinstance(f_alloc, opencl_mod.mem_alloc)
    assert isinstance(g_alloc, opencl_mod.mem_alloc)
    assert f_alloc.shape == g_alloc.shape
    assert f_alloc.dtype == g_alloc.dtype == np.float64
    assert len(f_alloc) == len(g_alloc) == 2
    return (clarray.dot(f_alloc[0], g_alloc[0]) +
            clarray.dot(f_alloc[1], g_alloc[1]))/config.K

def H_t_product(t, f_t, g_t):
    """ Computes the Hilbert space product between two elements in ``H_t``.

    ``H_t`` represents the Hilbert space at time ``t``. The implemented
    Hilbert space consists of the normalized real part of the complex dot
    product 


    Parameters
    ----------
    t : int
        Index of the referenced time sample. Takes values in 0,1,...,T. With
        (T+1) the total number of time samples.
    f_t, g_t : numpy.ndarray
        1-dimenisonal complex array representing an element of the Hilbert
        space at time ``t``, :math:`H_t`.

    Returns
    -------
    float

    Notes
    -----
    The computed formula is

    .. math::
        <f_t,g_t>_{H_t} = Re(<f_t, g_t>_{\\mathbb{C}})/dim(H_t)
        = Re( \\sum_k f_t(k)\\overline g_t(k))/dim(H_t)

    The Hilbert spaces must be real, meaning that the output of the inner
    product has to be a real number. In the implemented case here, we
    considered realified complex spaces.
    """
    assert checker.is_in_H_t(t, f_t) and checker.is_in_H_t(t, g_t)
    return np.real(np.dot(f_t, np.conj(g_t)))/H_DIMENSIONS[t]

def H_t_product_set_vector(t, f_t, g_t):
    """Hilbert space product between a set of elements vs a single one.

    An extension for fast evaluation between groups of elements.

    Parameters
    ----------
    t : int
        Index of the references time sample. Takes values in 0,1,...,T, where
        (T+1) the total number of time samples.
    f_t : numpy.ndarray
        (N,K) shaped complex array representing a collection of ``N``  elements
        of the Hilbert space at time ``t`` :math:`H_t` with dimension ``K``.
    g_t : numpy.ndarray
        1-dimensional complex array representing an element of the Hilbert
        space at time t :math:`H_t`.

    Returns
    -------
    numpy.ndarray
        (N,1)-dimensional float array with ``N`` the number of elements of the
        input collection ``f_t``.
    """
    assert checker.set_in_H_t(t, f_t) and checker.is_in_H_t(t, g_t)
    return np.real(np.dot(f_t, np.conj(g_t))).reshape(-1, 1)/H_DIMENSIONS[t]

def int_time_H_t_product(f, g):
    """Time integral of two collections of elements in each Hilbert space.

    A time integral in this context corresponds to the time average. Therefore
    this method computes the time average of the Hilbert space inner products.

    Parameters
    ----------
    f,g : list[numpy.ndarray]
        A list of size ``T``, where the ``t-th`` entry contains an element of
        the Hilbert space at time ``t``, :math:`H_t`.

    Returns
    -------
    float

    Notes
    -----
    Precisely, the computed value is

    .. math::
        \\sum_{t=0}^T w(t) <f_t, g_t>_{H_t}

    with :math:`w(t)` some weight at time :math:`t`, by default it is
    :math:`1/T`, with :math:`T` the total number of time samples.
    To change the way the spaces are weighted in this integral, modify
    ``config.time_weights``.
    """
    assert checker.is_in_H(f) and checker.is_in_H(g)
    output = 0
    time_weights = config.time_weights
    for t in range(config.T):
        output += time_weights[t]*H_t_product(t, f[t], g[t])
    return output

def K_t_cl(data_alloc, dummy_alloc): #data_real_cl, data_imag_cl,
           # mem_alloc_1, mem_alloc_2):
    """ Evaluation of pre-adjoint forward operator.

    Parameters
    ----------
    data_alloc: opencl_mod.mem_alloc
        a collection of 2 pyopencl.arrays, representing the real and imaginary
        part of elements belonging to the H space. 
        shape TTxK, with TT the total number of time smaples of the whole
        problem and K the dimension of each data space.  With dtype np.float64.
    dummy_alloc:
        allocated space in memory for computations. 
        It corresponds to 2 pyopencl.arrays of shape TxNxK, with T the 
        expected length of input times, N the expected number of points.
    Return
    ------
    callable[t_cl: pyopencl.array, x_cl: pyopencl.array, out_cl: pyopencl.array]
        t_cl is a 1 dimensional array of `int`s, representing the indexes of
        interest.  Its length is T.
        x_cl is a (N,2) shaped pyopencl.array representing N 2-dimensional
        points to evaluate to
        out_cl is TxN shaped pyopencl.array to write the solution to
    """
    assert isinstance(data_alloc, opencl_mod.mem_alloc)
    assert isinstance(dummy_alloc, opencl_mod.mem_alloc)
    assert data_alloc.dtype == dummy_alloc.dtype == np.float64
    assert len(data_alloc) == len(dummy_alloc) == 2
    assert data_alloc.shape[0] == config.T  # matching TT
    assert data_alloc.shape[1] == dummy_alloc.shape[2]   # matching K

    def funct(t_cl, x_cl, out_cl):
        assert t_cl.dtype == np.int32
        assert x_cl.dtype == out_cl.dtype == np.float64
        assert t_cl.shape[0] == dummy_alloc.shape[0] == out_cl.shape[0]  # T
        assert x_cl.shape[0] == out_cl.shape[1] == dummy_alloc.shape[1]  # N
        opencl_mod.TEST_FUNC(x_cl, t_cl, dummy_alloc)
        out_cl *= 0  # We set the values to zero, as einsum just adds up
        opencl_mod.einsum(t_cl, dummy_alloc[0], data_alloc[0], out_cl)
        opencl_mod.einsum(t_cl, dummy_alloc[1], data_alloc[1], out_cl)
        K = data_alloc.shape[1]
        out_cl /= K
    return funct


def K_t(t, f_t):
    """Evaluation of pre-adjoint forward operator of the inverse problem.

    Defines/evaluates the preadjoint of the forward operator at time sample
    ``t`` and element ``f`` of the t-th Hilbert space. The preadjoint maps into
    continuous functions.

    Parameters
    ----------
    t : int
        Index of the considered time sample. Takes values from 0,1,...,T-1
    f : numpy.ndarray
        2-dimensional complex array representing a member of the union of
        Hilbert spaces ``H_t``.

    Returns
    -------
    callable[numpy.ndarray, numpy.ndarray]
        function that takes (N,2)-sized arrays represnting ``N`` points in
        the domain Ω, and returns a (N,1)-sized array.

    Notes
    -----
    The preadjoint at time sample :math:`t` is a function that maps from the
    Hilbert space :math:`H_t` to the space of continuous functions on the
    domain :math:`C(\\Omega)`. The formula that defines this mapping is

    .. math::
        K_t(f_f) = x \\rightarrow <\\varphi(t,x), f_t>_{H_t}

    With :math:`\\varphi` the function ``TEST_FUNC`` input via
    :py:class:`src.DGCG.set_model_parameters`.
    """
    assert checker.is_valid_time(t) and checker.is_in_H(f_t)
    return lambda x: np.array([[H_t_product(t, f_t[t], test_func_j)
                                for test_func_j in TEST_FUNC(t, x)]]).T

def grad_K_t_cl(data_alloc, dummy_alloc):
    """ Evaluation of the gradient of the preadjoint forward operator.

    Parameters
    ----------
    data_alloc: opencl_mod.mem_alloc
        a collection of 2 pyopencl.arrays, representing the real and imaginary
        part of elements belonging to the H space. 
        shape TTxK, with TT the total number of time smaples of the whole
        problem and K the dimension of each data space.  With dtype np.float64.
    dummy_alloc:
        allocated space in memory for computations. 
        It corresponds to 4 pyopencl.arrays of shape TxNxK, with T the 
        expected length of input times, N the expected number of points.
    Return
    ------
    callable[t_cl: pyopencl.array, x_cl: pyopencl.array,
             out_alloc: opencl_mod.mem_alloc]
        t_cl is a 1 dimensional array of `int`s, representing the indexes of
        interest.  Its length is T.
        x_cl is a (N,2) shaped pyopencl.array representing N 2-dimensional
        points to evaluate to
        out_alloc correspond to a collection of two TxN shaped pyopencl arrays.
        to write the solution to.
    """
    assert isinstance(data_alloc, opencl_mod.mem_alloc)
    assert isinstance(dummy_alloc, opencl_mod.mem_alloc)
    assert data_alloc.dtype == dummy_alloc.dtype == np.float64
    assert len(data_alloc) == len(dummy_alloc) - 2 == 2
    assert data_alloc.shape[0] == config.T  # matching TT
    assert data_alloc.shape[1] == dummy_alloc.shape[2]   # matching K
    def funct(t_cl, x_cl, out_alloc):
        opencl_mod.GRAD_TEST_FUNC(x_cl, t_cl, dummy_alloc)
        out_alloc.amplify_by(0) # We set the values to zero, as einsum just adds up
        opencl_mod.einsum(t_cl, dummy_alloc[0], data_alloc[0], out_alloc[0])
        opencl_mod.einsum(t_cl, dummy_alloc[1], data_alloc[1], out_alloc[0],
                         wait_for=out_alloc[0].events)
        opencl_mod.einsum(t_cl, dummy_alloc[2], data_alloc[0], out_alloc[1])
        opencl_mod.einsum(t_cl, dummy_alloc[3], data_alloc[1], out_alloc[1],
                          wait_for=out_alloc[1].events)
        K = data_alloc.shape[1]
        out_alloc.amplify_by(1/K)  # Inner product normalization
    return funct

def grad_K_t(t, f):
    assert checker.is_valid_time(t) and checker.is_in_H(f)
    """Evaluation of the gradient of the preadjoint forward operator.

    Evaluates the gradient of the preadjoint of the forward operator at time
    sample ``t`` and element ``f`` of the t-th Hilbert space. The preadjoint
    maps into continuous functions.

    Parameters
    ----------
    t : int
        Index of the considered time sample. Takes values from 0,1,...,T, where
        (T+1) is the total number of time samples of the inverse problem.
    f : numpy.ndarray
        1-dimensional complex array representing a member of the t-th Hilbert
        space ``H_t``.

    Returns
    -------
    callable[numpy.ndarray, numpy.ndarray]
        function that takes (N,2)-sized arrays represnting ``N`` points in
        the domain Ω, and returns a (2,N,1)-sized array, with the first
        dimension corresponding to each partial derivative.

    Notes
    -----
    The gradient of the preadjoint at time sample :math:`t` is a function that
    maps from the Hilbert space :math:`H_t` to the space of continuous
    functions on the domain :math:`C(\\Omega)`. The formula that defines this
    mapping is

    .. math::
        K_t(f_f) = x \\rightarrow <\\nabla \\varphi(t,x), f_t>_{H_t}

    With :math:`\\varphi` the function ``TEST_FUNC`` input via
    :py:class:`src.DGCG.set_model_parameters` and the gradient taken
    in the spatial variables.

    This gradient is required to apply the gradient descent algorithm in
    the insertion step and sliding step.
    """
    return lambda x: np.array([H_t_product_set_vector(t, dxdy, f[t]) for
                               dxdy in GRAD_TEST_FUNC(t, x)])

def K_t_star_cl(t_cl, rho_cl, buff_alloc, output_alloc):
    """ Evaluation of forward operator for the inverse problem.

    Evaluates the forward operator at the given times t_cl and target 
    measure rho_cl. Returns a list of objects in the Hilbert space H_t

    Parameters
    ----------
    t_cl : pyopencl.array
        1-dimensional array of indexes, with dtype np.int32. The elements
        must be smaller than config.T
    rho_cl : classes.measure_cl
        A measure containing N curves/atoms.
    buff_alloc : opencl_mod.mem_alloc
        A container of 2 pyopencl.array of dimension TxNxK, with N the number
        of curves. An intermediate buffer for computations.
    output_alloc : opencl_mod.mem_alloc
        A container of 2 pyopencl.array of dimension TxK, representing the
        real and imaginary parts of the spaces H_t

    Returns
    -------
    None. output_buff gets overwritten.

    Notes
    -----
    Appears to be a useless function, stays idle. Incomplete
    """
    assert t_cl.shape[0] == output_alloc.shape[0] == buff_alloc.shape[0]  # T
    assert output_alloc.shape[1] == buff_alloc.shape[2] == config.K  # K
    assert buff_alloc[1] == rho_cl.weights.shape[0]  # N
    # First slice the curves from rho_cl, to correspond to the requested 
    # times in t_cl


    # For each of the given times, obtain the TxNxK matrix of φ(t, γ_i) evals.
    # these are allocated in the buff_alloc
    opencl_mod.TEST_FUNC_2(rho_cl.curves, t_cl, buff_alloc)
    pass


def K_t_star(t, rho):
    """Evaluation of forward operator of the inverse problem.

    Evaluates the forward operator at time
    sample ``t`` and measure ``rho``. The forward operator at time ``t``
    maps into the t-th Hilbert space ``H_t``.

    Parameters
    ----------
    t : int
        Index of the considered time sample. Takes values from 0,1,...,T, where
        (T+1) is the total number of time samples of the inverse problem.
    rho : :py:class:`src.classes.measure`
        Measure where the forward operator is evaluated.

    Returns
    -------
    numpy.ndarray
        1-dimensional complex array, representing an element of the t-th
        Hilbert space ``H_t``

    Notes
    -----
    The forward operator at time sample :math:`t` is a function that maps
    from the space of Radon measures :math:`\\mathcal{M}(\\Omega)` to the
    :math:`t`-th Hilbert space :math:`H_t`. The input measure of class
    :py:class:`src.classes.curve` is a dynamic measure, that once evaluated
    at time :math:`t`, becomes a Radon Measure.

    The formula that defines this function is the following Bochner integral

    .. math::
        K_t^*(\\rho_t) = \\int_{\\Omega} \\varphi(t,x) \\rho_t(dx)

    With :math:`\\varphi` the function ``TEST_FUNC`` input via
    :py:class:`src.DGCG.set_model_parameters`.
    """
    assert checker.is_valid_time(t) and isinstance(rho, classes.measure)
    return rho.spatial_integrate(t, lambda x: TEST_FUNC(t, x))

def K_t_star_full_cl(rho_cl, buff_alloc, output_alloc):
    """ Evaluation of forward operator for the inverse problem.

    Evaluates the forward operator at the given times t_cl and target 
    measure rho_cl. Returns a list of objects in the Hilbert space H_t

    Parameters
    ----------
    rho_cl : classes.measure_cl
        A measure containing N curves/atoms.
    buff_alloc : opencl_mod.mem_alloc
        A container of 2 pyopencl.array of dimension TxNxK, with N the number
        of curves. An intermediate buffer for computations.
    output_alloc : opencl_mod.mem_alloc
        A container of 2 pyopencl.array of dimension TxK, representing the
        real and imaginary parts of the spaces H_t

    Returns
    -------
    None. output_alloc gets overwritten.

    Notes
    -----
    """
    assert output_alloc.shape[0] == buff_alloc.shape[0] == config.T  # T
    assert output_alloc.shape[1] == buff_alloc.shape[2] == config.K  # K
    assert buff_alloc.shape[1] == rho_cl.weights.shape[0]  # N
    assert len(buff_alloc) == len(output_alloc) == 2

    # For each of the given times, obtain the TxNxK matrix of φ(t, γ_i) evals.
    # these are allocated in the buff_alloc
    opencl_mod.TEST_FUNC_3(rho_cl.curves, buff_alloc)
    # And we sum along the N dimesion weighted with the curve weights
    # the real and imaginary parts are summed separately
    opencl_mod.mat_vec_mul(rho_cl.weights, buff_alloc[0], output_alloc[0])
    opencl_mod.mat_vec_mul(rho_cl.weights, buff_alloc[1], output_alloc[1])

def K_t_star_full(rho):
    """Evaluation of forward operator of the inverse problem at all times.

    Evaluates the forward operator at all time
    samples and dynamic measure ``rho``. The output of this method is a list
    of elements in ``H_t``.

    Parameters
    ----------
    rho : :py:class:`src.classes.measure`
        Measure where the forward operator is evaluated.

    Returns
    -------
    list[numpy.ndarray]
        T-sized list of 1-dimensional complex arrays, representing elements
        of the Hilbert spaces ``H_t``

    Notes
    -----
    For further reference, see :py:meth:`src.operators.K_t_star`.
    """
    assert isinstance(rho, classes.measure)
    output = []
    for t in range(config.T):
        output.append(K_t_star(t, rho)[0])
    return np.array(output)

def overpenalization(s, M_0):
    """Overpenalization of the main inverse problem energy.

    Parameters
    ----------
    s, M_0 : float

    Returns
    -------
    float

    Notes
    -----
    This function is the one applied to the Benamou-Brenier energy when
    defining the surrogate linear problem described in the paper. It is a
    :math:`C^1` gluing of a linear and quadratic function.
    """
    assert isinstance(s, float) and isinstance(M_0, float)
    if s <= M_0:
        return s
    else:
        return (s**2 + M_0**2)/2/M_0

def main_energy(rho, f):
    """The main energy to minimize by the inverse problem.

    Parameters
    ----------
    measure : :py:class:`src.classes.measure`
        Radon measure.
    f : list[numpy.ndarray]
        list of elements of the Hilbert spaces ``H_t``

    Returns
    -------
    float

    Notes
    -----
    Implements the formula

    .. math::
        \\frac{1}{2T} \\sum_{t=0}^{T-1} || K_t^*(\\rho_t) - f_t ||_{H_t}^2 +
        J_{\\alpha, \\beta}(\\rho, m)

    Where :math:`m` is the momentum, that is implicitly defined for sparse
    measures as the ones used here.

    """
    assert isinstance(rho, classes.measure) and checker.is_in_H(f)
    # Computes the main energy, the one we seek to minimize
    # Input: rho ∈ M, a measure type object.
    #        f ∈ H, an element of H. 
    # Output: positive number
    forward = K_t_star_full(rho)
    diff = forward - f
    return int_time_H_t_product(diff, diff)/2 + sum(rho.weights)

