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

# Local imports
from . import config, checker
from . import classes

TEST_FUNC = None
GRAD_TEST_FUNC = None
H_DIMENSIONS = None


def H_t_product(t, f_t, g_t):
    """ Computes the Hilbert space product between two elements in ``H_t``.

    ``H_t`` represents the Hilbert space at time ``t``. The implemented
    Hilbert space consists of the normalized real part of the complex dot
    product 


    Parameters
    ----------
    t : int
        Index of the referenced time sample. Takes values in 0,1,...,T-1.
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
        Index of the references time sample. Takes values in 0,1,...,T-1
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
        1-dimensional complex array representing a member of the t-th Hilbert
        space ``H_t``.

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

def grad_K_t(t, f):
    assert checker.is_valid_time(t) and checker.is_in_H(f)
    """Evaluation of the gradient of the preadjoint forward operator.

    Evaluates the gradient of the preadjoint of the forward operator at time
    sample ``t`` and element ``f`` of the t-th Hilbert space. The preadjoint
    maps into continuous functions.

    Parameters
    ----------
    t : int
        Index of the considered time sample. Takes values from 0,1,...,T-1
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

def K_t_star(t, rho):
    """Evaluation of forward operator of the inverse problem.

    Evaluates the forward operator at time
    sample ``t`` and measure ``rho``. The forward operator at time ``t``
    maps into the t-th Hilbert space ``H_t``.

    Parameters
    ----------
    t : int
        Index of the considered time sample. Takes values from 0,1,...,T-1
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
    :py:class:`src.classes.curves` is a dynamic measure, that once evaluated
    at time :math:`t`, becomes a Radon Measure.

    The formula that defines this function is the following Bochner integral

    .. math::
        K_t^*(\\rho_t) = \\int_{\\Omega} \\varphi(t,x) \\rho_t(dx)

    With :math:`\\varphi` the function ``TEST_FUNC`` input via
    :py:class:`src.DGCG.set_model_parameters`.
    """
    assert checker.is_valid_time(t) and isinstance(rho, classes.measure)
    return rho.spatial_integrate(t, lambda x: TEST_FUNC(t, x))

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

