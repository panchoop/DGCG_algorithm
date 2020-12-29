"""Methods related to the problem's forward operator and Hilbert spaces."""
import numpy as np

# Local imports
from . import config, misc, checker
from . import classes

"""
 0. Time sampling
       The considered sampling times are to be defined in the config file,
       all the following spaces and methods are based on these time samples
       to be fixed.

1. The H_t spaces
       These spaces, for each time sample, could be different Hilbert spaces.
       This is why they have to be precisely defined, so each operator works
       properly. In the case of this code example, we will consider them all
       to be the complex space.
       Then, these H_t spaces are bundled together in a single space H.
       In the complex case, if H_t = C^{k_t}, then H = product_t C^{k_t}
       And such a Hilbert space can be embedded into H = C^{T, max_t(k_t)}
"""

# K is the dimension of each H_t. If needed, this dimension can be variable.
# defaul values

test_func = None
grad_test_func = None
K = None

def H_t_product(t, f_t, g_t):
    assert checker.is_in_H_t(t, f_t) and checker.is_in_H_t(t, g_t)
    # Computes the H_t product at between two elements in H_t
    # Input: t ∈ {0,1,...,T-1}
    #        f_t, g_t ∈ H_t = 1d numpy array with size K[t] 
    # Output: real value <f_t,g_t>_{H_t}.
    return np.real(np.dot(f_t, np.conj(g_t)))/K[t]

def H_t_product_set_vector(t, f_t, g_t):
    assert checker.set_in_H_t(t, f_t) and checker.is_in_H_t(t, g_t)
    # Computes the H_t product at between a numpy array of H_t elements with 
    # a sigle element in H_t
    # Input: t ∈ {0,1,...,T-1}
    #        f_t ∈ np.array of H_t elements,  g_t ∈ H_t = 1d numpy array
    # Output: Nx1 numpy array with values <f_t[i],g_t>_{H_t} for each H_t 
    #         element in H_t
    return np.real(np.dot(f_t, np.conj(g_t))).reshape(-1, 1)/K[t]

def int_time_H_t_product(f, g):
    assert checker.is_in_H(f) and checker.is_in_H(g)
    # Computes ∫<f_t, g_t>_{H_t} dt
    # Input : f,g ∈ H.
    # Output: real number.
    output = 0
    time_weights = config.time_weights
    for t in range(config.T):
        output += time_weights[t]*H_t_product(t, f[t], g[t])
    return output


"""
2. Forward operators K_t, K_t^*
        A simple way to define these bounded linear operators is to use
        convolutional kernels. In our case, we consider a family of test
        functions, whose sampling in certain locations in Fourier space,
        correspond to our H_t spaces.
        φ_t(x) is a continuous function from Ω to H_t, therefore
        K_t: H_t -> C(Ω); K_t(f_t) = <f_t,φ_t(x)>_{H_t}
        K_t_star: M(Ω) -> H_t; K_t(ρ) = <ρ, φ_t>_{M, C}

2.1 Sampling patterns
        To map into a Hilbert space that is manageable with a computer, we
        sample these φ_t(x) functions, leading to data in a complex C^K space.
"""

"""
2.2 Test functions and cut-off
        For the first order optimality conditions to be always valid, it is
        required to cut-off the boundary values.
"""

def K_t(t, f):
    assert checker.is_valid_time(t) and checker.is_in_H(f)
    # K_t: H_t -> C(\overline Ω)
    # K_t(f) = <f, φ(.,x)>_{H_t}
    # # It allows evaluation on a set of point x in Nx2 numpy array
    # Input: t ∈ {0,1,...,T-1}, f ∈ H.
    # Output: function x ∈ NxD -> Nx1, N is the number of eval. points.
    return lambda x: np.array([[H_t_product(t, f[t], test_func_j)
                                for test_func_j in test_func(t, x)]]).T

def grad_K_t(t, f):
    assert checker.is_valid_time(t) and checker.is_in_H(f)
    # ∇ K_t: H_t -> C(\overline Ω)
    # K_t(f) = <f, ∇φ(.,x)>_{H_t}
    # # It allows evaluation on a set of point x in Nx2 numpy array
    # Input: t ∈ {0,1,...,T-1}, f ∈ H.
    # Output: function x ∈ NxD -> 2xNx1, N number of eval. points, 2 for dx,dy.
    return lambda x: np.array([H_t_product_set_vector(t, dxdy, f[t]) for
                               dxdy in grad_test_func(t, x)])

def K_t_star(t, rho):
    assert checker.is_valid_time(t) and isinstance(rho, classes.measure)
    # K_t^*: M(Ω) -> H_t
    # K_t^*(ρ) = ρ_t(φ_t(·))
    # Input: t ∈ {0,1,...,T-1}, rho ∈ M, a measure.
    # Output: an element of H_t, an 1xK numpy array with K the dimension of H_t.
    return rho.spatial_integrate(t, lambda x: test_func(t, x))

def K_t_star_full(rho):
    assert isinstance(rho, classes.measure)
    # K_t_star: M(Ω) -> H
    # same as K_t_star, but it returns an element in H, not H_t
    # Input: rho ∈ M, a measure.
    # Output: an element of H, it is a numpy list of T elements, each a numpy
    #         array of size K[t], which is the dimension of H_t.
    output = []
    for t in range(config.T):
        output.append(K_t_star(t, rho)[0])
    return np.array(output)

def overpenalization(s, M_0):
    assert isinstance(s, float) and isinstance(M_0, float)
    # the φ(t) function that is applied to the Benamou-Brenier energy
    # Input: s, M_0 ∈ R, real numbers.
    # Output: real number 
    if s <= M_0:
        return s
    else:
        return (s**2 + M_0**2)/2/M_0

def main_energy(measure, f):
    assert isinstance(measure, classes.measure) and checker.is_in_H(f)
    # Computes the main energy, the one we seek to minimize
    # Input: measure ∈ M, a measure type object.
    #        f ∈ H, an element of H. 
    # Output: positive number
    forward = K_t_star_full(measure)
    diff = forward - f
    return int_time_H_t_product(diff, diff)/2 + sum(measure.weights)

